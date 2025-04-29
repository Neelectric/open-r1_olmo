### Written by Neel Rajani, 13.04.25
### Here we do analysis of sparsity across training

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

def normalized_update(before: torch.tensor, after: torch.tensor, epsilon=1e-10) -> np.array:
    before = before.detach().to(torch.float16).numpy()
    after = after.detach().to(torch.float16).numpy()
    diff = after - before
    
    # claude suggested a small epsilon to avoid division by zero
    norm_factor = np.maximum(np.abs(before), epsilon)
    norm_diff = diff / norm_factor
    return norm_diff



def compare_sparsity(base_model_id, ft_model_id, revision):
    """Compares the weights of a base model and a given checkpoint, using the normalized Frobenius norm of differences and standard deviations."""
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_id,
        # attn_implementation="flash_attention_2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ft_model_id,
        revision=revision,
        # attn_implementation="flash_attention_2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        cache_dir="data/"+ft_model_id,
    )
    # sanity check 
    base_params = dict(base_model.named_parameters())
    ckpt_params = dict(ckpt_model.named_parameters())
    assert base_params.keys() == ckpt_params.keys()
    
    total = len(base_params)
    torch.set_printoptions(precision=10)
    
    hist_counts = np.zeros(num_bins)
    total_elements = 0
    
    results_dict = {
        "q_proj": [],
        "k_proj": [],
        "v_proj": [],
        "o_proj": [],
        "gate_proj": [],
        "up_proj": [],
        "down_proj": []
    }
    print("ignoring embed and unembed, layernorms.")
    for name, base_param in tqdm(base_model.named_parameters(), dynamic_ncols=True, total=total, disable=False):
        ckpt_param = ckpt_params[name]
        name_list = name.split(".")
        if "layers" in name_list:
            if "self_attn" in name_list or "mlp" in name_list:
                diff_matrix = normalized_update(base_param, ckpt_param)
                abs_diff = np.abs(diff_matrix)
                curr_hist, _ = np.histogram(abs_diff, bins=bin_edges)
                hist_counts += curr_hist
                total_elements += diff_matrix.size
                
                # Free memory
                del diff_matrix, abs_diff
                        
    return hist_counts


if __name__ == "__main__":
    base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    num_bins = 100  # Adjust as needed
    bin_edges = np.logspace(-8, 0, num_bins+1)  # Log bins from 10^-8 to 1
    revisions = list_revisions(sft_model_id)
    print(revisions)
    for revision in tqdm(revisions, dynamic_ncols=True):
        hist_counts = compare_sparsity(base_model_id, sft_model_id, revision)
        np.save("data/" + sft_model_id + revision, hist_counts)
    
    # plot_histogram(sft_model_id, grpo_model_id)