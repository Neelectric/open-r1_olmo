from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from tqdm import tqdm
import torch
import torch.nn.functional as F
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


# def calc_kl(base_logits, ft_model_id, prompts, tokenizer, benchmark_id):
#     """Process all revisions for a given model"""
    
#     revisions = list_revisions(ft_model_id)
#     print(f"Found {len(revisions)} revisions for {ft_model_id}: {revisions}")
#     assert len(revisions) == 20   
    
#     base_probs = F.softmax(base_logits, dim=-1) # softmaxing in order to be able to compute KL
    
#     kls = []
#     for revision in tqdm(revisions, desc=f"Processing {ft_model_id}"):
#         pass
#         print(f"Loading checkpoint model: {ft_model_id}, revision: {revision}")
#         ft_logits = collect_logits(ft_model_id, prompts, tokenizer, benchmark_id, revision=revision, cache_dir=None, batch_size=10)
#         kl = F.kl_div(
#             F.log_softmax(ft_logits, dim=-1),
#             base_probs,
#             reduction='batchmean',
#             log_target=False
#             )
#         kls.append(kl)
#     return kls

def base_vs_ft(base_model, ft_model_id, prompts, tokenizer, benchmark_id, batch_size):
    save_path = "results/" + benchmark_id + "/" + ft_model_id
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    max_prompts = len(prompts)
    prompts = prompts[0:max_prompts]
    
    try:
        with open(save_path + f"/kls_{max_prompts}.json", "r") as f:
            kls_dict = json.load(f)
        print(f"Found and loaded kls for {ft_model_id} on {benchmark_id} with max_prompts {max_prompts}!")
        return kls_dict
    except:
        print(f"Did not find kls for {ft_model_id} on {benchmark_id}, under kls_{max_prompts}.json, computing them now...")
    
    
    revisions = list_revisions(ft_model_id)
    print(f"Found {len(revisions)} revisions for {ft_model_id}: {revisions}")
    if len(revisions) != 20:
        print(f"WARNING! THERE ARE ONLY {len(revisions)} revisions!")
    
    
    
    # prep inputs and hyperparams
    kls_dict = {}
    num_prompts = len(prompts)
    num_batches = math.ceil(num_prompts/batch_size)
    throwaway_input_ids = tokenizer(prompts, return_tensors="pt", padding="longest").input_ids
    print(f"our throwaway inputs had size {throwaway_input_ids.shape}")
    max_length = throwaway_input_ids.shape[1]
        
    # let's loop through all revisions one at a time
    for revision in tqdm(revisions, desc=f"Processing {ft_model_id}"):
        print(f"Loading in model {ft_model_id} revision {revision}")
        ft_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=ft_model_id,
                revision=revision,
                device_map="cuda:1",
                torch_dtype=torch.bfloat16,
                cache_dir="data/",
            )
        
        kls_at_rev = []
        
        for i in tqdm(range(num_batches), dynamic_ncols=True):
            # grab a batch, tokenize it to max length, make a copy, and send to base/ft models
            batch_prompts = prompts[i*batch_size : (i+1)*batch_size]
            # base_inputs = tokenizer(batch_prompts, return_tensors="pt", padding="max_length", max_length=max_length)
            base_inputs = tokenizer(batch_prompts, return_tensors="pt")
            # tqdm.write("not padding to max length but processing a prompt at a time out of fear of padding messing with predictions")
            ft_inputs = copy.deepcopy(base_inputs)
            
            base_inputs = base_inputs.to(base_model.device)            
            with torch.inference_mode():
                base_outputs = base_model(**base_inputs)
            base_logits = base_outputs.logits.detach().cpu()
            base_probs = F.softmax(base_logits, dim=-1) 
            
            ft_inputs = ft_inputs.to(ft_model.device)
            with torch.inference_mode():
                ft_outputs = ft_model(**ft_inputs)
            ft_logits = ft_outputs.logits.detach().cpu()
            ft_probs = F.softmax(ft_logits, dim=-1) 
            # tqdm.write("CAREFUL WTIH PAD TOKENS!!!!!!")
            
            kl = F.kl_div(
                torch.log(ft_probs),
                base_probs,
                reduction='batchmean',
                log_target=False
            )
            kls_at_rev.append(kl)
        kl_tensor = torch.stack(kls_at_rev, dim=0)
        mean_for_rev = torch.mean(kl_tensor)
        kls_dict[revision] = mean_for_rev.item()
        del ft_inputs, ft_outputs, ft_logits, ft_probs
        torch.cuda.empty_cache()
        del ft_model
        torch.cuda.empty_cache()
        print(kls_dict)
        
    print(kls_dict)
    with open(save_path + f"/kls_{max_prompts}.json", "w") as f:
            json.dump(kls_dict, f)
    # kls_tensor = torch.stack(kls, dim=0)
    # torch.save(kls_tensor, save_path + "/kls_tensor.pt")
    

    return kls_dict


# torch.distributions.kl.kl_divergence

def prepare_benchmark_prompts(tokenizer, benchmark_id):
    """Applies the chat template and tokenizes a huggingface benchmark dataset, returns list of formatted prompts in string format."""
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


# def collect_logits(model_id, prompts, tokenizer, benchmark_id, revision="main", cache_dir=None, batch_size=10):
#     """Returns the predictions of a model on a set of prompts."""    
#     save_path = "results/" + benchmark_id + "/" + model_id
#     Path(save_path).mkdir(parents=True, exist_ok=True)
#     try:
#         all_logits = torch.load(save_path +"/logits.pt", map_location="cpu", weights_only=True)
#         print(f"Found and loaded logits for {model_id} on {benchmark_id}!")
#     except:
#         print(f"Could not load logits for {model_id} on {benchmark_id}, computing them now...")
#         model = AutoModelForCausalLM.from_pretrained(
#                 pretrained_model_name_or_path=model_id,
#                 revision=revision,
#                 device_map="auto",
#                 torch_dtype=torch.bfloat16,
#                 cache_dir=cache_dir,
#             )
        
#         ### before we start this loop, we need to ensure we always pad to the same max size! unfortunately need to tokenize whole dataset first for this
#         throwaway_input_ids = tokenizer(prompts, return_tensors="pt", padding="longest").input_ids
#         print(f"our throwaway inputs had size {throwaway_input_ids.shape}")
#         max_length = throwaway_input_ids.shape[1]
        
#         num_prompts = len(prompts)
#         num_batches = math.ceil(num_prompts/batch_size)
#         all_logits = []
        
#         for i in tqdm(range(num_batches), dynamic_ncols=True):
#             batch_prompts = prompts[i*batch_size : (i+1)*batch_size]
#             inputs = tokenizer(batch_prompts, return_tensors="pt", padding="max_length", max_length=max_length).to(model.device)
#             with torch.inference_mode():
#                 outputs = model(**inputs)
#             batch_logits = outputs.logits.detach().cpu()
#             all_logits.append(batch_logits)
#             del inputs, outputs
#             if backend == "cuda":
#                 torch.cuda.empty_cache()
#             elif backend == "mps":
#                 torch.mps.empty_cache()       
#         logits_tensor = torch.cat(all_logits, dim=0) # going from list of num_batches tensors, to a tensor of shape [num_batches*bsz, num_toks, vocab_size]
#         torch.save(logits_tensor, save_path + "/logits.pt")
#         del model
#         if backend == "cuda":
#             torch.cuda.empty_cache()
#         elif backend == "mps":
#             torch.mps.empty_cache()
#     return logits_tensor


def plot_kls(grpo_model_id, grpo_kls_dict, sft_model_id, sft_kls_dict):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import pandas as pd
    import numpy as np
    
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Function to extract step numbers from revision strings
    def extract_step(revision):
        match = re.search(r'step-(\d+)', revision)
        if match:
            return int(match.group(1))
        return None
    
    # Process GRPO data
    grpo_steps = []
    grpo_kls = []
    for rev, kl in grpo_kls_dict.items():
        step = extract_step(rev)
        if step is not None:
            grpo_steps.append(step)
            # Convert tensor to float if needed
            kl_value = float(kl) if hasattr(kl, 'item') else float(kl)
            grpo_kls.append(kl_value)
    
    # Process SFT data
    sft_steps = []
    sft_kls = []
    for rev, kl in sft_kls_dict.items():
        step = extract_step(rev)
        if step is not None:
            sft_steps.append(step)
            # Convert tensor to float if needed
            kl_value = float(kl) if hasattr(kl, 'item') else float(kl)
            sft_kls.append(kl_value)
    
    # Sort by steps to ensure proper line plotting
    grpo_data = sorted(zip(grpo_steps, grpo_kls))
    sft_data = sorted(zip(sft_steps, sft_kls))
    
    # Unzip for plotting
    grpo_steps_sorted, grpo_kls_sorted = zip(*grpo_data) if grpo_data else ([], [])
    sft_steps_sorted, sft_kls_sorted = zip(*sft_data) if sft_data else ([], [])
    
    # Plot
    plt.plot(grpo_steps_sorted, grpo_kls_sorted, 'o-', label='GRPO', linewidth=2, markersize=8)
    plt.plot(sft_steps_sorted, sft_kls_sorted, 'o-', label='SFT', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('KL Divergence (base||fine-tuned)', fontsize=14)
    plt.title('KL Divergence Between Base and Fine-tuned Models During Training', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show step numbers clearly
    plt.ticklabel_format(style='plain', axis='x')
    
    # Save figure
    plt.tight_layout()
    grpo_name = grpo_model_id.split("_")[1]
    sft_name = sft_model_id.split("_")[1]
    
    plt.savefig(f'figures/kls/{grpo_name}_vs_{sft_name}.pdf', dpi=300)
    plt.show()

def process_all_models():
    """Process and compare all training regimes"""
    base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.14"
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv02.00"
    
    # base_model_id = "google/gemma-2-2b-it"
    
    # base_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    # grpo_model_id = "toastloaf/smollm2-135m-it-orca-agentinstruct-creative"
    
    benchmark_id = "HuggingFaceH4/MATH-500"
    
    tokenizer = AutoTokenizer.from_pretrained(grpo_model_id, padding_side="left")    
    print(f"tokenizer comes from {grpo_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=base_model_id,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
            )
    prompts = prepare_benchmark_prompts(tokenizer, benchmark_id)
    
    
    grpo_kls_dict = base_vs_ft(base_model=base_model, ft_model_id=grpo_model_id, prompts=prompts, tokenizer=tokenizer, benchmark_id=benchmark_id, batch_size=1)
    sft_kls_dict = base_vs_ft(base_model=base_model, ft_model_id=sft_model_id, prompts=prompts, tokenizer=tokenizer, benchmark_id=benchmark_id, batch_size=1)
    
    plot_kls(grpo_model_id, grpo_kls_dict, sft_model_id, sft_kls_dict)
    
    # compare_training_regimes(training_regimes, bin_edges)
    

if __name__ == "__main__":
    process_all_models()