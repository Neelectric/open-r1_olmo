










if __name__ == "__main__":
    sft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
    grpo_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
    
    compare_norm_trajectories(sft_model_id, grpo_model_id)
    
    plot_histogram(sft_model_id, grpo_model_id)