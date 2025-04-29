from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from tqdm import tqdm
import torch
import json
import fire
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.animation import PillowWriter
from utils import list_revisions
import re
import pandas as pd

def normalized_update(before, after, epsilon=1e-10):
    """Calculate normalized difference between tensors"""
    # Stay in PyTorch for as long as possible
    diff = after - before
    norm_factor = torch.maximum(before.abs(), torch.tensor(epsilon, device=before.device, dtype=before.dtype))
    norm_diff = (diff / norm_factor).to(torch.float16)
    # Only convert to numpy at the end
    return norm_diff.detach().numpy()

def compare_sparsity(base_model_id, ft_model_id, revision):
    """Compare sparsity between base model and fine-tuned checkpoint"""
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Loading checkpoint model: {ft_model_id}, revision: {revision}")
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ft_model_id,
        revision=revision,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        cache_dir="data/"+ft_model_id,
    )
    
    # Sanity check 
    base_params = dict(base_model.named_parameters())
    ckpt_params = dict(ckpt_model.named_parameters())
    assert base_params.keys() == ckpt_params.keys()
    
    # Process parameters in batches
    hist_counts = np.zeros(num_bins)
    total_elements = 0
    
    for name, base_param in tqdm(base_model.named_parameters(), desc="Processing parameters", dynamic_ncols=True):
        # Skip parameters that don't match our criteria
        if "layers" not in name or ("self_attn" not in name and "mlp" not in name):
            continue
            
        ckpt_param = ckpt_params[name]
        
        # Calculate normalized difference
        diff_matrix = normalized_update(base_param, ckpt_param)
        abs_diff = np.abs(diff_matrix)
        
        # Update histogram
        curr_hist, _ = np.histogram(abs_diff, bins=bin_edges)
        hist_counts += curr_hist
        total_elements += diff_matrix.size
        
        # Free memory
        del diff_matrix, abs_diff
        
        # Occasionally run garbage collection
        if total_elements % 10000000 == 0:  # Every ~10M elements
            import gc
            gc.collect()
    
    print(f"Processed {total_elements} total elements")
    return hist_counts

def combine_histograms(histogram_data):
    """Combine histogram data across all revisions"""
    combined_hist = None
    
    for revision, hist_counts in histogram_data.items():
        if combined_hist is None:
            combined_hist = hist_counts.copy()
        else:
            combined_hist += hist_counts
    
    return combined_hist

def plot_combined_histogram(combined_hist, bin_edges, model_id, output_dir="plots"):
    """Plot combined histogram of parameter changes across all training steps"""
    plt.figure(figsize=(12, 8))
    
    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Normalize histogram counts to get density
    total_counts = np.sum(combined_hist)
    if total_counts > 0:
        normalized_counts = combined_hist / total_counts
    else:
        normalized_counts = combined_hist
    
    # Plot as a line with log-log scale
    plt.loglog(bin_centers, normalized_counts, linewidth=2)
    
    # Add grid and labels
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Create title
    model_short = model_id.split('/')[-1].split('_')[0]
    model_type = model_id.split('/')[-1].split('_')[-1]  # SFT vs GRPO part
    plt.title(f"Distribution of Parameter Changes Across All Training\n{model_short} {model_type}", fontsize=16)
    
    # Add statistics annotation
    non_zero_changes = combined_hist[1:]  # Skip the first bin (which contains zero changes)
    non_zero_sum = np.sum(non_zero_changes)
    if non_zero_sum > 0:
        mean_bin = np.sum(bin_centers[1:] * non_zero_changes) / non_zero_sum
        text = f"Mean change: {mean_bin:.2e}"
        plt.annotate(text, xy=(0.7, 0.9), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_short}_{model_type}_combined_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

def compare_training_regimes(training_regimes, bin_edges, output_dir="plots"):
    """Compare histograms between different training regimes"""
    plt.figure(figsize=(12, 8))
    
    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Plot each training regime
    for model_id, combined_hist in training_regimes.items():
        # Normalize histogram counts to get density
        total_counts = np.sum(combined_hist)
        if total_counts > 0:
            normalized_counts = combined_hist / total_counts
        else:
            normalized_counts = combined_hist
        
        # Get label from model ID
        model_short = model_id.split('/')[-1]
        if "SFT" in model_short:
            label = "SFT"
            color = "blue"
        elif "GRPO" in model_short:
            label = "GRPO"
            color = "red"
        else:
            label = model_short
            color = None
        
        # Plot as a line with log-log scale
        plt.loglog(bin_centers, normalized_counts, linewidth=2, label=label, color=color)
    
    # Add grid and labels
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Comparison of Parameter Changes Across Training Regimes\nHigher density at lower values = more sparse updates", fontsize=16)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/training_regime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def calculate_sparsity_metrics(combined_hist, bin_edges):
    """Calculate sparsity metrics from histogram data"""
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Calculate percentage of changes below certain thresholds
    total_counts = np.sum(combined_hist)
    
    metrics = {}
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for threshold in thresholds:
        # Find bins below threshold
        bins_below = bin_centers < threshold
        count_below = np.sum(combined_hist[bins_below])
        percent_below = (count_below / total_counts) * 100 if total_counts > 0 else 0
        metrics[f"percent_below_{threshold:.0e}"] = percent_below
    
    # Calculate mean and median parameter change
    non_zero_bins = combined_hist > 0
    if np.any(non_zero_bins):
        weighted_values = bin_centers[non_zero_bins] * combined_hist[non_zero_bins]
        mean_change = np.sum(weighted_values) / np.sum(combined_hist[non_zero_bins])
        metrics["mean_change"] = mean_change
        
        # Approximate median (finding the bin where cumulative sum crosses 50%)
        cumsum = np.cumsum(combined_hist)
        median_idx = np.searchsorted(cumsum, total_counts / 2)
        if median_idx < len(bin_centers):
            metrics["median_change"] = bin_centers[median_idx]
    
    return metrics

def process_all_models():
    """Process and compare all training regimes"""
    base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    global num_bins, bin_edges
    num_bins = 100  # Adjust as needed
    bin_edges = np.logspace(-8, 0, num_bins+1)  # Log bins from 10^-8 to 1
    
    # Process SFT model
    sft_histograms = process_model(base_model_id, sft_model_id)
    sft_combined = combine_histograms(sft_histograms)
    plot_combined_histogram(sft_combined, bin_edges, sft_model_id)
    
    # Process GRPO model
    grpo_histograms = process_model(base_model_id, grpo_model_id)
    grpo_combined = combine_histograms(grpo_histograms)
    plot_combined_histogram(grpo_combined, bin_edges, grpo_model_id)
    
    # Compare training regimes
    training_regimes = {
        sft_model_id: sft_combined,
        grpo_model_id: grpo_combined
    }
    compare_training_regimes(training_regimes, bin_edges)
    
    # Calculate and print sparsity metrics
    print("\nSparsity Metrics:")
    print("=" * 50)
    
    sft_metrics = calculate_sparsity_metrics(sft_combined, bin_edges)
    grpo_metrics = calculate_sparsity_metrics(grpo_combined, bin_edges)
    
    print("SFT Model:")
    for key, value in sft_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nGRPO Model:")
    for key, value in grpo_metrics.items():
        print(f"  {key}: {value:.6f}")

def process_model(base_model_id, model_id):
    """Process all revisions for a given model"""
    histograms = {}
    
    # List revisions
    revisions = list_revisions(model_id)
    print(f"Found {len(revisions)} revisions for {model_id}: {revisions}")
    
    for revision in tqdm(revisions, desc=f"Processing {model_id}"):
        # Construct output file path
        model_short = model_id.split('/')[-1].split('_')[0]
        model_type = model_id.split('/')[-1].split('_')[-1]
        
        # Make sure the directory exists
        results_dir = f"results/{model_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = f"{results_dir}/{revision}.npy"
        
        # Check if the file exists
        if os.path.exists(output_file):
            print(f"Loading existing data for {revision}")
            hist_counts = np.load(output_file)
            histograms[revision] = hist_counts
        else:
            print(f"Processing revision: {revision}")
            hist_counts = compare_sparsity(base_model_id, model_id, revision)
            histograms[revision] = hist_counts
            
            # Save results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, hist_counts)
            
            # Clear memory
            import gc
            gc.collect()
    
    return histograms

if __name__ == "__main__":
    process_all_models()