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
from datasets import Dataset, load_dataset
import copy
from pathlib import Path

import math

if torch.backends.mps.is_available():
    backend = "mps"
elif torch.cuda.is_available():
    backend = "cuda" 
else:
    backend = "cpu"


def calc_kl(base_logits, ft_model_id, prompts, tokenizer, benchmark_id):
    """Process all revisions for a given model"""
    
    revisions = list_revisions(ft_model_id)
    print(f"Found {len(revisions)} revisions for {ft_model_id}: {revisions}")
    assert len(revisions) == 20   
    
    kl_divergences = []
    for revision in tqdm(revisions, desc=f"Processing {ft_model_id}"):
        pass
        print(f"Loading checkpoint model: {ft_model_id}, revision: {revision}")
        logits = collect_logits(ft_model_id, prompts, tokenizer, benchmark_id, revision=revision, cache_dir=None, bsz=10)
        
        
        
         # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )
        
        # del ckpt_model
        # torch.cuda.empty_cache()
        
    return kl_divergences

# torch.distributions.kl.kl_divergence

def prepare_benchmark_prompts(tokenizer, benchmark_id):
    """Applies the chat template and tokenizes a huggingface benchmark dataset."""
    ds = load_dataset(benchmark_id)["test"]
    conversations = []
    for elt in ds:
        conversations.append([
            {"role": "user", "content": elt["problem"]},
            {"role": "assistant", "content": elt["solution"]}
        ])

    conv_ds = Dataset.from_dict({"chat": conversations})
    templated_ds = conv_ds.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)}, remove_columns=["chat"])
    templated_list = [elt["formatted_chat"] for elt in templated_ds]
    return templated_list

def del_and_flush_cache(delete_me):
    del delete_me
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()
    return

def collect_logits(model_id, prompts, tokenizer, benchmark_id, revision="main", cache_dir=None, bsz=10):
    """Returns the predictions of a model on a set of prompts."""    
    save_path = "results/" + benchmark_id + "/" + model_id
    Path(save_path).mkdir(parents=True, exist_ok=True)
    try:
        logits = torch.load(save_path +"/logits.pt", weights_only=True)
    except:
        print(f"Could not load logits for {model_id} on {benchmark_id}, computing them now")
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                revision=revision,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            )
        num_prompts = len(prompts)
        num_batches = math.ceil(num_prompts/bsz)
        # output_list = []
        logits = []
        print("HARDCODING NUMBATCHES TO 2")
        num_batches = 2
        for i in tqdm(range(num_batches), dynamic_ncols=True):
            batch_prompts = prompts[i*bsz : (i+1)*bsz]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs)
            # output_list.append(outputs)
            logits.append(outputs.logits)
            del_and_flush_cache(outputs)
        torch.save(logits, save_path + "/logits.pt")
        # print(output_list)
        del_and_flush_cache(model)
    return logits

def process_all_models():
    """Process and compare all training regimes"""
    # base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    base_model_id = "google/gemma-2-2b-it"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.14"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv02.00"
    benchmark_id = "HuggingFaceH4/MATH-500"
    
    # tokenizer = AutoTokenizer.from_pretrained(grpo_model_id, padding_side="left")
    ptyhia_tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
    print(f"tokenizer comes from {base_model_id}")
    prompts = prepare_benchmark_prompts(ptyhia_tokenizer, benchmark_id)

    base_logits = collect_logits(model_id=base_model_id, prompts=prompts, tokenizer=ptyhia_tokenizer, benchmark_id=benchmark_id)
    
    # KL (base, GRPO_ckpt) for all ckpts
    # grpo_kls = calc_kl(base_logits=base_logits, ft_model_id=grpo_model_id, prompts=prompts, tokenizer=tokenizer, benchmark_id=benchmark_id)
    
    # KL (base, sft_ckpt) for all ckpts
    # sft_kls = calc_kl(base_logits=base_logits, ft_model_id=sft_model_id, prompts=prompts, tokenizer=tokenizer, benchmark_id=benchmark_id)
    
    # # Compare training regimes
    # training_regimes = {
    #     sft_model_id: sft_combined,
    #     grpo_model_id: grpo_combined
    # }
    # compare_training_regimes(training_regimes, bin_edges)
    

if __name__ == "__main__":
    process_all_models()