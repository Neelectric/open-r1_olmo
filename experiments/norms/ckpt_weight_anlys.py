### Written by Neel Rajani, 13.04.25
### Here we do analysis of a model's weights across training

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from transformers import AutoModelForCausalLM

from tqdm import tqdm
import torch
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import imageio.v2 as imageio
from experiments.utils import list_revisions
import re
import pandas as pd


def compare_base_and_ckpt(base_model, ft_model_id, revision):
  """Compares the weights of a base model and a given checkpoint, using the normalized Frobenius norm of differences and standard deviations."""
  ckpt_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=ft_model_id,
    revision=revision,
    # attn_implementation="flash_attention_2",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    cache_dir="data/",
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
    "gate_proj": [],
    "up_proj": [],
    "down_proj": []
  }
  print("ignoring embed and unembed, layernorms.")
  for name, base_param in tqdm(base_model.named_parameters(), dynamic_ncols=True, total=total, disable=False):
    ckpt_param = ckpt_params[name]
    name_list = name.split(".")
    if ("layers" in name_list) and ("q_norm" not in name_list) and ("k_norm" not in name_list):
            
      frob_norm_base = torch.linalg.norm(base_param)
      frob_norm_diff = torch.linalg.norm(ckpt_param - base_param)
      normed_frob_norm_diff = frob_norm_diff / frob_norm_base if frob_norm_base > 0 else float('inf')
      
      # print("normed_frob_norm_diff", normed_frob_norm_diff)   
      diff = torch.abs(ckpt_param - base_param)
      mean = torch.mean(diff)
      # print("mean", mean, "\n")
      if ("self_attn" in name_list or "mlp" in name_list):
        results_dict[name_list[4]].append(normed_frob_norm_diff.item())
      
  return results_dict


def plot_results(results_dict, ft_model_id, revision, vmin, vmax, percentage):
  full_model_name = ft_model_id.split("/")[1]
  model_name = full_model_name.split("_")[0]
  method = full_model_name.split("_")[1]
  method_no_version = method.split("v")[0]
  save_path = f"results/{ft_model_id}/plot_results.pdf"
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
  data = np.array([results_dict[key] for key in results_dict.keys()]).T

  x_labels = list(results_dict.keys()) 
  y_labels = [f"Layer {i}" for i in range(data.shape[0])]
  
  # Set font sizes
  title_size = 28
  label_size = 24
  tick_size = 20
  colorbar_size = 20
  
  fig, ax = plt.subplots(figsize=(14, 12))
  
  # Create the heatmap with increased font sizes
  heatmap = sns.heatmap(
      data, 
      annot=False, 
      cmap="viridis", 
      xticklabels=x_labels, 
      yticklabels=y_labels, 
      ax=ax, 
      vmin=vmin, 
      vmax=vmax,
      cbar_kws={
          'label': f'Norm difference',
          'format': '%.5f'
      }
  )
  plt.tight_layout()
  fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.07)
  
  # Increase font size for labels and title
  ax.set_xlabel("Parameters", fontsize=label_size)
  ax.set_ylabel("Layers", fontsize=label_size)
  ax.set_title(
      f"Normalized frobenius norm of the differences for each matrix:\n{model_name} after {percentage}% of {method_no_version}", 
      fontsize=title_size
  )
  
  # Increase font size for tick labels
  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  
  # Get colorbar and modify its appearance
  cbar = ax.collections[0].colorbar
  cbar.ax.set_ylabel('Norm difference',
                    fontsize=colorbar_size)
  cbar.ax.tick_params(labelsize=colorbar_size)
  ax.invert_yaxis()  # this should layer 0 is at the bottom?
  return fig

def checkpoints_to_percentages(results_dicts:dict) -> list:
  """Extracts step numbers from checkpoint names and returns sorted percentages."""
  def extract_step(checkpoint):
      match = re.search(r'step-(\d+)', checkpoint)
      return int(match.group(1)) if match else 0

  # Sort checkpoints by step number and exclude 'main' checkpoint
  sorted_checkpoints = sorted([cp for cp in results_dicts.keys() if cp != 'main'], key=extract_step)
  sorted_steps = [extract_step(cp) for cp in sorted_checkpoints]
  
  # Calculate the max step to determine 100% completion
  max_step = max(sorted_steps) if sorted_steps else 1  # Avoid division by zero
  
  # Create a list of percentages
  sorted_percentages = [int((step / max_step) * 100) for step in sorted_steps]
  print(f"Max step found to be {max_step}")
  print(f"hence, percentages are {sorted_percentages}")
  return sorted_percentages

def plot_trajectories(results_dicts: dict, gif_dir:str) -> None:
  """Written by Claude 3.7 Sonnet, plots the trajectories of norms throughout training"""
  
  def extract_step(checkpoint):
      match = re.search(r'step-(\d+)', checkpoint)
      return int(match.group(1)) if match else 0
  sorted_checkpoints = sorted([cp for cp in results_dicts.keys() if cp != 'main'], key=extract_step)
  sorted_steps = [extract_step(cp) for cp in sorted_checkpoints]
  max_step = max(sorted_steps) if sorted_steps else 1  # Avoid division by zero
  sorted_percentages = [(step / max_step) * 100 for step in sorted_steps]

  # Define the matrices to analyze
  matrices = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

  # Set up the plotting environment
  sns.set(style="whitegrid")
  sns.set_palette("viridis", 32)  # Palette with 32 colors for layers
  
  # Increase all font sizes
  plt.rcParams.update({
      'font.size': 16,
      'axes.titlesize': 24,
      'axes.labelsize': 20,
      'xtick.labelsize': 16,
      'ytick.labelsize': 16,
      'legend.fontsize': 16,
      'legend.title_fontsize': 18
  })

  # Create individual plots for each matrix
  for matrix in matrices:
      plt.figure(figsize=(15, 8))
      
      # Prepare data
      data = []
      for checkpoint in sorted_checkpoints:
          step = extract_step(checkpoint)
          percentage = (step / max_step) * 100  # Convert to percentage
          
          if matrix in results_dicts[checkpoint]:
              layer_norms = results_dicts[checkpoint][matrix]
              
              for layer, norm in enumerate(layer_norms):
                  data.append({
                      'Step': percentage,  # Use percentage instead of raw step
                      'Layer': f'Layer {layer}',
                      'Norm': norm
                  })
      
      df = pd.DataFrame(data)
      if not df.empty:
          # Create the plot
          sns.lineplot(
              x='Step', 
              y='Norm', 
              hue='Layer', 
              data=df
          )
          
          plt.title(f'Norm Trajectory for {matrix}', fontsize=24, pad=20)
          plt.xlabel('Training Progress (%)', fontsize=20, labelpad=15)  # Updated label
          plt.ylabel('Frobenius Norm', fontsize=20, labelpad=15)
          
          # Move legend outside the plot with larger font size
          plt.legend(title='Layer', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=18)
          plt.tight_layout()
          plt.savefig(f'{gif_dir}/norm_trajectory_{matrix}.png', dpi=300, bbox_inches='tight')
          plt.close()

  # Also create a summary plot showing average norm across layers for each matrix
  plt.figure(figsize=(15, 8))

  # Define a distinct color palette for different matrix types
  distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

  for i, matrix in enumerate(matrices):
      # Calculate mean norm across layers for each checkpoint
      mean_norms = []
      
      for checkpoint in sorted_checkpoints:
          if matrix in results_dicts[checkpoint]:
              layer_norms = results_dicts[checkpoint][matrix]
              if layer_norms:
                  mean_norms.append(np.mean(layer_norms))
              else:
                  mean_norms.append(np.nan)
          else:
              mean_norms.append(np.nan)
      
      # Plot the mean norms with distinct colors, using percentages on x-axis
      plt.plot(sorted_percentages, mean_norms, marker='o', linewidth=2, label=matrix, color=distinct_colors[i])

  plt.title('Mean Norm Trajectory Across Layers', fontsize=24, pad=20)
  plt.xlabel('Training Progress (%)', fontsize=20, labelpad=15)  # Updated label
  plt.ylabel('Mean Frobenius Norm', fontsize=20, labelpad=15)
  plt.legend(title='Matrix Type', fontsize=16, title_fontsize=18)
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f'{gif_dir}/mean_norm_trajectories.png', dpi=300)

  print("Visualization complete! Individual plots for each matrix saved as 'norm_trajectory_*.png'")
  print("Summary plot saved as 'mean_norm_trajectories.png'")
  return

def revision_processed(results_dicts: dict, revision: str) -> bool:
  """Checks if the results dict contains computations for every matrix."""
  results_dict_revision = results_dicts[revision]
  for key,value in results_dict_revision.items():
    if len(value) != 32:
      return False
  return True

def norm_comparison(base_model_id:str, ft_model_id:str) -> None:
  """Compares each revision (ie. checkpoint) of the fine-tuned model to the base model. Produces heatmap plots of this, collates them into a gif, and plots projectories per matrix type."""
  revisions = list_revisions(ft_model_id)
  print(revisions)
  
  results_dicts = {}
  counter = 0
  gif_dir = f"results/{ft_model_id}"
  os.makedirs(gif_dir, exist_ok=True)
  json_path = gif_dir + "/results_dict.json"
  try:
    with open(json_path) as f:
      results_dicts = json.load(f)
    print("successfully loaded results dicts, will resume here")
  except:
    print("results_dicts not found for this fine-tune, starting to populate it")
    
  base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    # attn_implementation="flash_attention_2",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
  )
  
  # compares each revision to the base model via normalised frobenius norm and save results
  for revision in tqdm(revisions, dynamic_ncols=True):
    if revision == "v02.08_1epoch-step-000000152":
      continue
    elif revision == "v02.08-step-000000456":
      continue
    print("*"*100)
    print(f"NOW COMPARING TO REVISION {revision}")
    if results_dicts.get(revision):
        print("we have an entry for this revision, checking if it was completed...")
        if revision_processed(results_dicts=results_dicts, revision=revision):
          print("it was fully processed! skipping...")
          continue
    print("we don't have a (complete) entry yet, starting comparison")
    results_dict = compare_base_and_ckpt(base_model, ft_model_id, revision)
    print("not including q/k norms cuz idc about them")
    results_dicts[revision] = results_dict
    with open(json_path, 'w') as f:
      json.dump(results_dicts, f)
      print(f"cached results for revision {revision}!")
    
  # finds the global mins and maxes to plot on the same colourplot scales
  min_max_source = "final"
  global_min = 99999
  global_max = -99999
  if min_max_source == "global":
    for revision in revisions:
      for key, value in results_dicts[revision].items():
        matrix_max = max(value)
        matrix_min = min(value)
        if matrix_max > global_max:
          global_max = matrix_max
        if matrix_min < global_min:
          global_min = matrix_min
    print(f"GLOBAL MIN IS {global_min} AND GLOBAL MAX IS {global_max}")
  elif min_max_source == "final":
    last_rev = revisions[-1]
    for key, value in results_dicts[last_rev].items():
        matrix_max = max(value)
        matrix_min = min(value)
        if matrix_max > global_max:
          global_max = matrix_max
        if matrix_min < global_min:
          global_min = matrix_min
    print(f"FINAL MIN IS {global_min} AND FINAL MAX IS {global_max}")
      
  # plots the results for each checkpoint
  num_revisions = len(revisions)
  interval_size = int(100//num_revisions)
  percentages = [i for i in range(interval_size, 100+interval_size, interval_size)]
  print(f"we have {num_revisions} revisions, so our percentage intervals are {percentages}")
  figs = []
  for i, revision in enumerate(revisions):
    if revision == "v02.08_1epoch-step-000000152":
      continue
    elif revision == "v02.08-step-000000456":
      continue
    percentage_through_training = percentages[i]
    fig = plot_results(results_dicts[revision], ft_model_id, revision, global_min, global_max, percentage=percentage_through_training)
    figs.append(fig)
  
  # saves each figure as separate PNG file
  png_paths = []
  for i, fig in enumerate(figs):
    png_path = f"{gif_dir}/frame_{i}.png"
    fig.savefig(png_path, dpi=100)
    png_paths.append(png_path)
    plt.close(fig)
  
  # Use PIL to create a GIF from the PNG files
  images = [imageio.imread(png_path) for png_path in png_paths]
  gif_path = f"{gif_dir}/training_dynamics.gif"
  imageio.mimsave(gif_path, images, fps=4, loop=0)
  print(f"GIF saved to {gif_path}")
  
  ## quick and dirty script to plot trajectories too
  plot_trajectories(results_dicts, gif_dir)


if __name__ == '__main__':
  base_model_id = "allenai/OLMo-2-1124-7B-Instruct"
  # ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.14"
  ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv02.08"
  
  # base_model_id = "Qwen/Qwen2.5-Math-7B"
  # ft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  norm_comparison(base_model_id=base_model_id, ft_model_id=ft_model_id)