import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_norm_trajectories(sft_model_id, grpo_model_id, output_dir="figures/comparison_figures"):
    """
    Compares norm trajectories between SFT and GRPO models, plotting both on the same graph
    with different line styles but the same colors for the same matrix types.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths to results dictionaries
    sft_results_path = f"figures/{sft_model_id}/results_dict.json"
    grpo_results_path = f"figures/{grpo_model_id}/results_dict.json"
    
    # Load results dictionaries
    with open(sft_results_path, 'r') as f:
        sft_results = json.load(f)
    
    with open(grpo_results_path, 'r') as f:
        grpo_results = json.load(f)
    
    # Extract step numbers from checkpoint names
    def extract_step(checkpoint):
        match = re.search(r'step-(\d+)', checkpoint)
        return int(match.group(1)) if match else 0
    
    # Sort checkpoints for both models (excluding 'main')
    sft_checkpoints = sorted([cp for cp in sft_results.keys() if cp != 'main'], key=extract_step)
    grpo_checkpoints = sorted([cp for cp in grpo_results.keys() if cp != 'main'], key=extract_step)
    
    sft_steps = [extract_step(cp) for cp in sft_checkpoints]
    grpo_steps = [extract_step(cp) for cp in grpo_checkpoints]
    
    # Calculate max steps and percentages
    sft_max_step = max(sft_steps) if sft_steps else 1
    grpo_max_step = max(grpo_steps) if grpo_steps else 1
    
    sft_percentages = [(step / sft_max_step) * 100 for step in sft_steps]
    grpo_percentages = [(step / grpo_max_step) * 100 for step in grpo_steps]
    
    # Define the matrices to analyze
    matrices = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Define a distinct color palette for different matrix types
    distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
    
    # Set up the plotting environment - increase font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18
    })
    
    # Create the summary plot
    plt.figure(figsize=(15, 10))
    
    # Plot data for both models
    for i, matrix in enumerate(matrices):
        # Calculate mean norms for SFT model
        sft_mean_norms = []
        for checkpoint in sft_checkpoints:
            if matrix in sft_results[checkpoint]:
                layer_norms = sft_results[checkpoint][matrix]
                if layer_norms:
                    sft_mean_norms.append(np.mean(layer_norms))
                else:
                    sft_mean_norms.append(np.nan)
            else:
                sft_mean_norms.append(np.nan)
        
        # Calculate mean norms for GRPO model
        grpo_mean_norms = []
        for checkpoint in grpo_checkpoints:
            if matrix in grpo_results[checkpoint]:
                layer_norms = grpo_results[checkpoint][matrix]
                if layer_norms:
                    grpo_mean_norms.append(np.mean(layer_norms))
                else:
                    grpo_mean_norms.append(np.nan)
            else:
                grpo_mean_norms.append(np.nan)
        
        # Plot SFT data with dashed lines
        plt.plot(sft_percentages, sft_mean_norms, 
                 linestyle='--', marker='o', linewidth=2, 
                 color=distinct_colors[i], 
                 label=f"{matrix} (SFT)")
        
        # Plot GRPO data with solid lines
        plt.plot(grpo_percentages, grpo_mean_norms, 
                 linestyle='-', marker='s', linewidth=2.5, 
                 color=distinct_colors[i], 
                 label=f"{matrix} (GRPO)")
    
    # Customize the plot
    plt.title('Mean Norm Trajectory Comparison: SFT vs GRPO', fontsize=24, pad=20)
    plt.xlabel('Training Progress (%)', fontsize=20, labelpad=15)
    plt.ylabel('Mean Frobenius Norm', fontsize=20, labelpad=15)
    plt.grid(True)
    
    # Add a legend with custom styling
    legend = plt.legend(title="Matrix Type & Model", 
                      fontsize=16, title_fontsize=18,
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sft_vs_grpo_norm_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {output_dir}/sft_vs_grpo_norm_comparison.png")
    
    # Return the plot figure for further customization if needed
    return plt.gcf()

if __name__ == "__main__":
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    compare_norm_trajectories(sft_model_id, grpo_model_id)