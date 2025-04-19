# MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# MODEL=Neelectric/Qwen2.5-7B-Instruct_GRPOv00.05
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL=CohereForAI/c4ai-command-r7b-12-2024
# MODEL=nvidia/AceInstruct-7B
# MODEL=allenai/OLMo-2-1124-7B-Instruct
# MODEL=Neelectric/Qwen2.5-7B-Instruct_SFTv00.13
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.12

NUM_GPUS=8
MAX_TOKENS=4096
MAX_GPU_MEM_USAGE=0.8

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$MAX_TOKENS,max_num_batched_tokens=$MAX_TOKENS,gpu_memory_utilization=$MAX_GPU_MEM_USAGE,generation_parameters={max_new_tokens:$MAX_TOKENS,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,gpu_memory_utilization=$MAX_GPU_MEM_USAGE,generation_parameters={max_new_tokens:$MAX_TOKENS,temperature:0.6,top_p:0.95}"
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_TOKENS,max_num_batched_tokens=$MAX_TOKENS,gpu_memory_utilization=$MAX_GPU_MEM_USAGE,generation_parameters={max_new_tokens:$MAX_TOKENS,temperature:0.6,top_p:0.95}"

# AIME 2024
# TASK=aime24
lighteval vllm $MODEL_ARGS  "lighteval|aime24|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

# ifeval
lighteval vllm $MODEL_ARGS "extended|ifeval|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# # GSM8k
lighteval vllm $MODEL_ARGS "lighteval|gsm8k|5|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# # # MMLU
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
lighteval vllm $MODEL_ARGS "leaderboard|mmlu|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# MMMLU-Pro 
# leaderboard_mmlu_pro
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=auto, \
    --tasks leaderboard_mmlu_pro \
    --output_path $OUTPUT_DIR \
    --batch_size 16 \
    --apply_chat_template 