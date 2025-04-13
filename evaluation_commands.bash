# MODEL=Neelectric/SmolLM2-1.7B-Instruct_GRPO
# MODEL=Neelectric/SmolLM2-1.7B-Instruct_GRPO
# MODEL=Qwen/Qwen2.5-Math-1.5B
# MODEL=Neelectric/Qwen2.5-7B-Instruct_GRPOv00.05
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL=CohereForAI/c4ai-command-r7b-12-2024
# MODEL=nvidia/AceInstruct-7B
MODEL=allenai/OLMo-2-1124-7B-Instruct
# MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.08
# MODEL=Qwen/Qwen2.5-32B
NUM_GPUS=8
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95,}"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.9,generation_parameters={\"max_new_tokens\":4096,\"temperature\":0.6,\"top_p\":0.95}"
# system_prompt="You are a helpful assistant."
OUTPUT_DIR=data/evals/$MODEL

# # # AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # # # MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # # # GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # # # # LiveCodeBench
# # # # lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
# # # #     --use-chat-template \
# # # #     --output-dir $OUTPUT_DIR 

# # # ifeval
# lighteval vllm $MODEL_ARGS "extended|ifeval|0|1" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR


# # # GSM8k
# lighteval vllm $MODEL_ARGS "lighteval|gsm8k|5|1" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # # # MMLU
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
# lighteval vllm $MODEL_ARGS "leaderboard|mmlu|0|1" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR


# MMMLU-Pro 
# leaderboard_mmlu_pro
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS \
    --tasks leaderboard_mmlu_pro \
    --apply_chat_template \
    --output_path $OUTPUT_DIR
    # --batch_size auto
