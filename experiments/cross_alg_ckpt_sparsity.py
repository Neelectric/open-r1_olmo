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

def normalized_update_batch(before_tensors, after_tensors, epsilon=1e-10):
    """Process multiple tensors at once to avoid repeated conversions"""
    all_diffs = []
    
    for before, after in zip(before_tensors, after_tensors):
        # Stay in PyTorch for as long as possible
        diff = after - before
        norm_factor = torch.maximum(before.abs(), torch.tensor(epsilon, device=before.device, dtype=before.dtype))
        norm_diff = diff / norm_factor if norm_factor > 0 else torch.inf
        
        # Only convert to numpy at the end
        all_diffs.append(norm_diff.cpu().numpy().flatten())
    
    # Concatenate all flattened differences
    if all_diffs:
        return np.concatenate(all_diffs)
    return np.array([])

def compare_sparsity(base_model_id, ft_model_id, revision, batch_size=10):
    """Optimized version that processes parameters in batches"""
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
    
    # Filter parameter names based on criteria
    param_names = [
        name for name in base_params.keys() 
        if "layers" in name and ("self_attn" in name or "mlp" in name)
    ]
    print(f"Processing {len(param_names)} parameter tensors")
    
    # Process parameters in batches
    hist_counts = np.zeros(num_bins)
    total_elements = 0
    
    for i in tqdm(range(0, len(param_names), batch_size), desc="Processing parameter batches"):
        batch_names = param_names[i:i+batch_size]
        
        # Collect tensors
        base_tensors = [base_params[name] for name in batch_names]
        ckpt_tensors = [ckpt_params[name] for name in batch_names]
        
        # Process batch
        diff_batch = normalized_update_batch(base_tensors, ckpt_tensors)
        
        if diff_batch.size > 0:
            abs_diff = np.abs(diff_batch)
            curr_hist, _ = np.histogram(abs_diff, bins=bin_edges)
            hist_counts += curr_hist
            total_elements += diff_batch.size
        
        # Explicitly clear tensors to free memory
        del base_tensors, ckpt_tensors, diff_batch
        if 'abs_diff' in locals():
            del abs_diff
        
        # Force garbage collection occasionally
        if i % (batch_size * 5) == 0:
            import gc
            gc.collect()
    
    print(f"Processed {total_elements} total elements")
    return hist_counts

if __name__ == "__main__":
    base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    num_bins = 100  # Adjust as needed
    bin_edges = np.logspace(-8, 0, num_bins+1)  # Log bins from 10^-8 to 1
    
    # List revisions only once
    revisions = list_revisions(sft_model_id)
    print(f"Found {len(revisions)} revisions: {revisions}")
    
    for revision in tqdm(revisions, desc="Processing revisions"):
        output_file = f"data/{sft_model_id}{revision}.npy"
        
        # Skip if already processed
        if os.path.exists(output_file):
            print(f"Skipping {revision} - already processed")
            continue
            
        print(f"Processing revision: {revision}")
        hist_counts = compare_sparsity(base_model_id, sft_model_id, revision)
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, hist_counts)
        
        # Clear memory between revisions
        import gc
        gc.collect()
    
    print("All revisions processed successfully!")