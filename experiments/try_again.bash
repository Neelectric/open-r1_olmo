NUM_GPUS=8
NUM_TOKS=4096
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL


# ### AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# # MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

#### LiveCodeBench
####lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
####    --use-chat-template \
 ###   --output-dir $OUTPUT_DIR 

# ifeval
lighteval vllm $MODEL_ARGS "extended|ifeval|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# # GSM8k
lighteval vllm $MODEL_ARGS "lighteval|gsm8k|5|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# # # # MMLU
lighteval vllm $MODEL_ARGS "leaderboard|mmlu|5|0" \
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


lighteval vllm "pretrained=Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "lighteval|gsm8k|5|0" \
    --use-chat-template 