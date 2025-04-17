### By Neel Rajani, 13.04.25
### Here we do analysis of a model's weights across training

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from tqdm import tqdm
import torch
import fire
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.animation import PillowWriter




### ok what is it we want in here
# i want to give an hf model name
# and get an analysis of the weights changing across training as a result
# so probably a 'figures' folder right
# into which we paste graphs which plot changes across steps


        
        
def list_revisions(model_id: str) -> list[str]:
  """Returns all revisions of a model from the hub."""
  api = HfApi()
  refs = api.list_repo_refs(model_id)
  branch_names = [branch.name for branch in refs.branches]
  revisions = branch_names[:0:-1] 
  return revisions + ["main"]

# def compare_base_and_ckpt(base_model_id: str, ft_model_id: str, checkpoint_id: str) -> dict:
#   """Compares the weights of a base model and a given checkpoint, using the normalized Frobenius norm of differences and standard deviations."""
#   base_model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=base_model_id,
#     attn_implementation="flash_attention_2",
#     device_map="cpu",
#     torch_dtype=torch.bfloat16,
#   )
#   ckpt_model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=ft_model_id,
#     revision=checkpoint_id,
#     attn_implementation="flash_attention_2",
#     device_map="cpu",
#     torch_dtype=torch.bfloat16,
#   )
#   # sanity check 
#   base_params = dict(base_model.named_parameters())
#   ckpt_params = dict(ckpt_model.named_parameters())
#   assert base_params.keys() == ckpt_params.keys()
  
#   comparison_results = {}
#   total = len(base_params)
#   for name, base_param in tqdm(base_model.named_parameters(), dynamic_ncols=True, total=total):
#     ckpt_param = ckpt_params[name]
#     norm_of_diff = torch.linalg.norm(ckpt_param - base_param).item()
#     norm_of_base = torch.linalg.norm(base_param).item()
#     rel_diff = norm_of_diff / norm_of_base if norm_of_base > 0 else float('inf')
#     comparison_results[name] = {
#       "frob_norm_of_diff": norm_of_diff,
#       "frob_norm_of_rel_diff": rel_diff,
#       'param_size': base_param.numel()
#     }
#     # assert norm_of_diff == 0.0
#   return comparison_results

# def sort_by_diff_norm(comparison_results, metric='frob_norm_of_rel_diff'):
#     """
#     Sort the comparison results by the specified metric in descending order.
    
#     Args:
#         comparison_results: Dictionary output from compare_checkpoints function
#         metric: Which metric to sort by ('absolute_diff_norm', 'relative_diff_norm', etc.)
        
#     Returns:
#         List of tuples (parameter_name, values) sorted by the specified metric
#     """
#     sorted_results = sorted(
#         comparison_results.items(), 
#         key=lambda item: item[1][metric], 
#         reverse=True
#     )
#     return sorted_results
  
# def aggregate_by_matrix_type(comparison_results):
#     """
#     Aggregate comparison results by matrix type (queries, keys, values, etc.)
    
#     Args:
#         comparison_results: Dictionary output from compare_checkpoints function
        
#     Returns:
#         Dictionary with aggregated statistics by matrix type
#     """
#     # Initialize data structure to hold aggregated results
#     aggregated = defaultdict(lambda: {
#         'total_absolute_diff': 0.0,
#         'total_parameters': 0,
#         'max_absolute_diff': 0.0,
#         'max_relative_diff': 0.0,
#         'param_names': []
#     })
    
#     # Define patterns to identify matrix types
#     matrix_patterns = {
#         'query_proj': ['q_proj', 'self_attn.q_proj', 'query_proj'],
#         'key_proj': ['k_proj', 'self_attn.k_proj', 'key_proj'],
#         'value_proj': ['v_proj', 'self_attn.v_proj', 'value_proj'],
#         'output_proj': ['o_proj', 'self_attn.o_proj', 'output_proj'],
#         'mlp_gate': ['gate_proj', 'mlp.gate_proj'],
#         'mlp_up': ['up_proj', 'mlp.up_proj'],
#         'mlp_down': ['down_proj', 'mlp.down_proj'],
#         'layer_norm': ['norm', 'ln_', 'layer_norm'],
#         'embedding': ['embed', 'embedding'],
#         'lm_head': ['lm_head']
#     }
    
#     # Process each parameter
#     for name, values in comparison_results.items():
#         # Find the matrix type
#         matrix_type = 'other'
#         for type_name, patterns in matrix_patterns.items():
#             if any(pattern in name for pattern in patterns):
#                 matrix_type = type_name
#                 break
        
#         # Update aggregated statistics
#         aggregated[matrix_type]['total_absolute_diff'] += values['frob_norm_of_diff']
#         aggregated[matrix_type]['total_parameters'] += values['param_size']
#         aggregated[matrix_type]['max_absolute_diff'] = max(
#             aggregated[matrix_type]['max_absolute_diff'], 
#             values['frob_norm_of_diff']
#         )
#         aggregated[matrix_type]['max_relative_diff'] = max(
#             aggregated[matrix_type]['max_relative_diff'], 
#             values['frob_norm_of_rel_diff']
#         )
#         aggregated[matrix_type]['param_names'].append(name)
    
#     # Calculate average change per parameter for each type
#     for matrix_type in aggregated:
#         if aggregated[matrix_type]['total_parameters'] > 0:
#             aggregated[matrix_type]['avg_change_per_param'] = (
#                 aggregated[matrix_type]['total_absolute_diff'] / 
#                 aggregated[matrix_type]['total_parameters']
#             )
    
#     return aggregated

# def print_matrix_type_summary(aggregated_results):
#     """
#     Print a summary of matrix type changes sorted by total change
#     """
#     # Sort by total absolute difference
#     sorted_types = sorted(
#         aggregated_results.items(),
#         key=lambda x: x[1]['total_absolute_diff'],
#         reverse=True
#     )
    
#     print("\nMatrix Type Change Summary (sorted by total absolute difference):")
#     print("=" * 80)
#     print(f"{'Matrix Type':<15} {'Total Abs Diff':<15} {'Total Params':<15} {'Avg Change/Param':<15}")
#     print("-" * 80)
    
#     for matrix_type, stats in sorted_types:
#         print(f"{matrix_type:<15} {stats['total_absolute_diff']:<15.6f} "
#               f"{stats['total_parameters']:<15} {stats.get('avg_change_per_param', 0):<15.8f}")

def compare_base_and_ckpt(base_model_id, ft_model_id, revision):
  """Compares the weights of a base model and a given checkpoint, using the normalized Frobenius norm of differences and standard deviations."""
  base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    attn_implementation="flash_attention_2",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
  )
  ckpt_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=ft_model_id,
    revision=revision,
    attn_implementation="flash_attention_2",
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
  
  results_dict = {
    "q_proj": [],
    "k_proj": [],
    "v_proj": [],
    "o_proj": [],
    "q_norm": [],
    "k_norm": [],
    "gate_proj": [],
    "up_proj": [],
    "down_proj": []
  }
  print("ignoring embed and unembed, layernorms.")
  for name, base_param in tqdm(base_model.named_parameters(), dynamic_ncols=True, total=total, disable=False):
    ckpt_param = ckpt_params[name]
    name_list = name.split(".")
    if "layers" in name_list:
      # print(name)
      # print(base_param.shape)
      # print(base_param)
      # print(ckpt_param)
      
      frob_norm_base = torch.linalg.norm(base_param)
      frob_norm_diff = torch.linalg.norm(ckpt_param - base_param)
      normed_frob_norm_diff = frob_norm_diff / frob_norm_base if frob_norm_base > 0 else float('inf')
      
      # print("normed_frob_norm_diff", normed_frob_norm_diff)   
      diff = torch.abs(ckpt_param - base_param)
      mean = torch.mean(diff)
      # print("mean", mean, "\n")
      if "self_attn" in name_list or "mlp" in name_list:
        results_dict[name_list[4]].append(normed_frob_norm_diff.item())
      
  return results_dict


def plot_results(results_dict, ft_model_id, revision, vmin, vmax):
  save_path = f"figures/{ft_model_id}/plot_results.pdf"
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
  data = np.array([results_dict[key] for key in results_dict.keys()]).T

  x_labels = list(results_dict.keys()) 
  y_labels = [f"Layer {i}" for i in range(data.shape[0])]
  fig, ax = plt.subplots(figsize=(12, 8))
  sns.heatmap(data, annot=False, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels, ax=ax, vmin=vmin, vmax=vmax)

  # Add labels and title
  ax.set_xlabel("Parameters")
  ax.set_ylabel("Layers")
  ax.set_title(f"Normalized frobenius norm of the differences for each matrix:\n{ft_model_id} vs. {revision}")
  ax.invert_yaxis()  # this should layer 0 is at the bottom?
  return fig


def main():
  # base_model_id = "Qwen/Qwen2.5-7B-Instruct"
  # ft_model_id = "Neelectric/Qwen2.5-7B-Instruct_SFTv00.13"
  
  base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
  # ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.09"
  ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10"
  revisions = list_revisions(ft_model_id)
  print(revisions)
  
  results_dicts = []
  counter = 0
  
  # lets compare each revision to the base model via normalised frobenius norm and save results
  for revision in tqdm(revisions, dynamic_ncols=True):
    print("*"*100)
    print(f"NOW COMPARING TO REVISION {revision}")
    print("*"*100)
    # comparison_results = compare_base_and_ckpt(base_model_id, ft_model_id, revision)
    # sorted_results = sort_by_diff_norm(comparison_results)
    # # for name,val in sorted_results:
    # #   print(name, val["frob_norm_of_rel_diff"])
      
    # aggregated_results = aggregate_by_matrix_type(comparison_results)
    # print_matrix_type_summary(aggregated_results)
    results_dict = compare_base_and_ckpt(base_model_id, ft_model_id, revision)
    print("removing q/k norms cuz idc")
    del results_dict["q_norm"]
    del results_dict["k_norm"]
    results_dicts.append(results_dict)
    
  # lets find the global mins and maxes to plot on the same colourplot scales
  global_min = 99999
  global_max = -9999
  for result_dict in results_dicts:
    for key, value in results_dict.items():
      matrix_max = max(value)
      matrix_min = min(value)
      if matrix_max > global_max:
        global_max = matrix_max
      if matrix_min < global_min:
        global_min = matrix_min
  print(f"GLOBAL MIN IS {global_min} AND GLOBAL MAX IS {global_max}")
      
  figs = []
  for i, revision in enumerate(revisions):
    fig = plot_results(results_dicts[i], ft_model_id, revision, global_min, global_max)
    figs.append(fig)

  gif_dir = f"figures/{ft_model_id}"
  os.makedirs(gif_dir, exist_ok=True)
  
  # Save each figure as a separate PNG file
  png_paths = []
  for i, fig in enumerate(figs):
    png_path = f"{gif_dir}/frame_{i}.png"
    fig.savefig(png_path, dpi=100)
    png_paths.append(png_path)
    plt.close(fig)
  
  # Use PIL to create a GIF from the PNG files
  
  import imageio.v2 as imageio
  images = [imageio.imread(png_path) for png_path in png_paths]
  gif_path = f"{gif_dir}/training_dynamics.gif"
  imageio.mimsave(gif_path, images, duration=0.1, loop=0)
  print(f"GIF saved to {gif_path}")
  
  # # Optionally, clean up the PNG files
  # for png_path in png_paths:
  #   os.remove(png_path)


if __name__ == '__main__':
  main()