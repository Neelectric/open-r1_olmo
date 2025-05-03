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
from experiments.utils import list_revisions
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

def plot_single_histogram(hist_counts, bin_edges, revision, model_id, output_dir="plots"):
    """Plot a single histogram for a specific revision"""
    plt.figure(figsize=(12, 8))
    
    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Normalize histogram counts to get density
    total_counts = np.sum(hist_counts)
    if total_counts > 0:
        normalized_counts = hist_counts / total_counts
    else:
        normalized_counts = hist_counts
    
    # Plot as a line with log-log scale
    plt.loglog(bin_centers, normalized_counts, linewidth=2)
    
    # Add grid and labels
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Make the revision more readable for the title
    revision_title = revision.replace("v01.05-step-", "step ")
    plt.title(f"Distribution of Parameter Changes for {model_id.split('/')[-1]}\n{revision_title}", fontsize=16)
    
    # Add statistics annotation
    non_zero_changes = hist_counts[1:]  # Skip the first bin (which contains zero changes)
    non_zero_sum = np.sum(non_zero_changes)
    if non_zero_sum > 0:
        mean_bin = np.sum(bin_centers[1:] * non_zero_changes) / non_zero_sum
        text = f"Mean change: {mean_bin:.2e}"
        plt.annotate(text, xy=(0.7, 0.9), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Ensure directory exists (create all needed directories)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simpler filename
    model_short = model_id.split('/')[-1].split('_')[0]  # Just use OLMo part
    step_match = re.search(r'step-(\d+)', revision)
    if step_match:
        step_num = step_match.group(1)
        filename = f"{model_short}_step{step_num}_histogram.png"
    else:
        safe_revision = revision.replace("/", "_")
        filename = f"{model_short}_{safe_revision}_histogram.png"
    
    # Save the figure
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()

def plot_all_histograms(histogram_data, bin_edges, model_id, output_dir="plots"):
    """Plot all histograms together for comparison"""
    plt.figure(figsize=(15, 10))
    
    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Get colors from a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(histogram_data)))
    
    # Sort revisions by step number
    sorted_revs = sorted(histogram_data.keys(), 
                         key=lambda x: int(re.search(r'step-(\d+)', x).group(1)) 
                         if re.search(r'step-(\d+)', x) else 0)
    
    # Plot each histogram
    for i, revision in enumerate(sorted_revs):
        # Normalize histogram counts to get density
        hist_counts = histogram_data[revision]
        total_counts = np.sum(hist_counts)
        if total_counts > 0:
            normalized_counts = hist_counts / total_counts
        else:
            normalized_counts = hist_counts
        
        # Extract step number for the label
        step_num = re.search(r'step-(\d+)', revision)
        if step_num:
            label = f"Step {int(step_num.group(1)):,}"
        else:
            label = revision
        
        plt.loglog(bin_centers, normalized_counts, linewidth=2, color=colors[i], label=label)
    
    # Add grid and labels
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(f"Distribution of Parameter Changes Across Training\n{model_id.split('/')[-1]}", fontsize=16)
    
    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Set axis limits
    plt.xlim(bin_edges[0], bin_edges[-1])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_id.split('/')[-1].split('_')[0]  # Just use OLMo part
    plt.savefig(f"{output_dir}/{model_short}_all_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create a heatmap version showing evolution over training
    create_histogram_evolution_heatmap(histogram_data, bin_edges, model_id, output_dir)

def create_histogram_evolution_heatmap(histogram_data, bin_edges, model_id, output_dir="plots"):
    """Create a heatmap showing how the histogram evolves over training steps"""
    # Sort revisions by step number
    sorted_revs = sorted(histogram_data.keys(), 
                         key=lambda x: int(re.search(r'step-(\d+)', x).group(1)) 
                         if re.search(r'step-(\d+)', x) else 0)
    
    # Extract step numbers for y-axis labels
    step_numbers = []
    for rev in sorted_revs:
        step_match = re.search(r'step-(\d+)', rev)
        if step_match:
            step_numbers.append(int(step_match.group(1)))
        else:
            step_numbers.append(0)
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(sorted_revs), len(bin_edges)-1))
    for i, rev in enumerate(sorted_revs):
        # Normalize histogram counts
        hist_counts = histogram_data[rev]
        total_counts = np.sum(hist_counts)
        if total_counts > 0:
            heatmap_data[i, :] = hist_counts / total_counts
        else:
            heatmap_data[i, :] = hist_counts
    
    # Apply log transformation for better visualization
    heatmap_data = np.log10(heatmap_data + 1e-10)  # Add small value to avoid log(0)
    
    # Create heatmap
    plt.figure(figsize=(16, 10))
    
    # Calculate bin centers for x-axis labels
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap="viridis", 
                xticklabels=[f"{x:.1e}" for x in bin_centers[::10]], 
                yticklabels=[f"{step:,}" for step in step_numbers],
                cbar_kws={'label': 'Log Density'})
    
    plt.xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    plt.ylabel("Training Step", fontsize=14)
    plt.title(f"Evolution of Parameter Changes During Training\n{model_id.split('/')[-1]}", fontsize=16)
    
    # Adjust x-axis ticks for readability
    plt.xticks(np.arange(0, len(bin_centers), 10), rotation=45)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_id.split('/')[-1].split('_')[0]  # Just use OLMo part
    plt.savefig(f"{output_dir}/{model_short}_histogram_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_animation(histogram_data, bin_edges, model_id, output_dir="plots"):
    """Create an animation showing how the histogram evolves over training steps"""
    # Sort revisions by step number
    sorted_revs = sorted(histogram_data.keys(), 
                         key=lambda x: int(re.search(r'step-(\d+)', x).group(1)) 
                         if re.search(r'step-(\d+)', x) else 0)
    
    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a writer
    writer = PillowWriter(fps=2)
    
    # Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_id.split('/')[-1].split('_')[0]  # Just use OLMo part
    animation_path = f"{output_dir}/{model_short}_histogram_animation.gif"
    
    # Set up x and y limits (using max across all histograms)
    max_density = 0
    for revision in sorted_revs:
        hist_counts = histogram_data[revision]
        total_counts = np.sum(hist_counts)
        if total_counts > 0:
            normalized_counts = hist_counts / total_counts
            max_density = max(max_density, np.max(normalized_counts))
    
    # Set up axes with log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(1e-6, max_density * 1.1)
    
    # Set labels
    ax.set_xlabel("Normalized Parameter Change (absolute)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Create animation
    with writer.saving(fig, animation_path, dpi=100):
        for i, revision in enumerate(sorted_revs):
            # Clear previous frame
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_ylim(1e-6, max_density * 1.1)
            ax.set_xlabel("Normalized Parameter Change (absolute)", fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            
            # Normalize histogram counts
            hist_counts = histogram_data[revision]
            total_counts = np.sum(hist_counts)
            if total_counts > 0:
                normalized_counts = hist_counts / total_counts
            else:
                normalized_counts = hist_counts
            
            # Plot histogram
            ax.plot(bin_centers, normalized_counts, linewidth=2)
            
            # Extract step number for the title
            step_match = re.search(r'step-(\d+)', revision)
            if step_match:
                step_num = int(step_match.group(1))
                ax.set_title(f"Distribution of Parameter Changes at Step {step_num:,}\n{model_id.split('/')[-1]}", fontsize=16)
            else:
                ax.set_title(f"Distribution of Parameter Changes for {revision}\n{model_id.split('/')[-1]}", fontsize=16)
            
            # Save frame
            writer.grab_frame()
    
    print(f"Animation saved to {animation_path}")

if __name__ == "__main__":
    base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    num_bins = 100  # Adjust as needed
    bin_edges = np.logspace(-8, 0, num_bins+1)  # Log bins from 10^-8 to 1
    
    # Dictionary to store histogram data for all revisions
    all_histograms = {}
    
    # List revisions only once
    revisions = list_revisions(sft_model_id)
    print(f"Found {len(revisions)} revisions: {revisions}")
    
    for revision in tqdm(revisions, desc="Processing revisions"):
        output_file = f"results/{sft_model_id}/{revision}.npy"
        
        # Check if the file exists
        if os.path.exists(output_file):
            print(f"Loading existing data for {revision}")
            hist_counts = np.load(output_file)
            all_histograms[revision] = hist_counts
            
            # Plot individual histogram
            plot_single_histogram(hist_counts, bin_edges, revision, sft_model_id)
        else:
            print(f"Processing revision: {revision}")
            hist_counts = compare_sparsity(base_model_id, sft_model_id, revision)
            all_histograms[revision] = hist_counts
            
            # Save results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, hist_counts)
            
            # Plot individual histogram
            plot_single_histogram(hist_counts, bin_edges, revision, sft_model_id)
            
            # Clear memory between revisions
            import gc
            gc.collect()
    
    # Plot all histograms together
    if all_histograms:
        print("Creating combined plots...")
        plot_all_histograms(all_histograms, bin_edges, sft_model_id)
        
        print("Creating animation...")
        create_animation(all_histograms, bin_edges, sft_model_id)
    
    print("All processing and plotting completed successfully!")