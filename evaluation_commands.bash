# MODEL=Neelectric/SmolLM2-1.7B-Instruct_GRPO
# MODEL=Neelectric/SmolLM2-1.7B-Instruct_GRPO
# MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFT
# MODEL=CohereForAI/c4ai-command-r7b-12-2024
MODEL=nvidia/AceInstruct-7B
NUM_GPUS=3
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# # # AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # # MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# # # GPQA Diamond
# # TASK=gpqa:diamond
# # lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
# #     --custom-tasks src/open_r1/evaluate.py \
# #     --use-chat-template \
# #     --output-dir $OUTPUT_DIR

# # # # LiveCodeBench
# # # lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
# # #     --use-chat-template \
# # #     --output-dir $OUTPUT_DIR 

# # # GSM8k
# # TASK=leaderboard|gsm8k
# lighteval vllm $MODEL_ARGS "lighteval|gsm8k|5|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # MMLU
# TASK=leaderboard|mmlu
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilization=0.75,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
lighteval vllm $MODEL_ARGS "leaderboard|mmlu|0|1" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
