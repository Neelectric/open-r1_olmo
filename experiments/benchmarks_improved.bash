uv pip uninstall transformers
uv pip uninstall vllm
uv pip uninstall flash-attn
uv pip install vllm==0.7.2
# transformers==4.51.3
#  + vllm==0.7.2
# python3; import nltk; nltk.download('punkt_tab')

NUM_GPUS=8
NUM_TOKS=4096
MAX_GPU_USAGE=0.8
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv02.00
# MODEL=Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.14
REVISION=main
# REVISION=v01.14-step-000000319
MODEL_ARGS="pretrained=$MODEL,revision=$REVISION,dtype=auto,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=$MAX_GPU_USAGE,max_num_batched_tokens=$NUM_TOKS,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL


### AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
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

# ifeval
lighteval vllm $MODEL_ARGS "extended|ifeval|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# # # GSM8k
lighteval vllm $MODEL_ARGS "lighteval|gsm8k|5|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MAX_GPU_USAGE=0.8
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=$MAX_GPU_USAGE,max_num_batched_tokens=4096,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"

# # # MMLU
lighteval vllm $MODEL_ARGS "leaderboard|mmlu|5|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


## ## MMMLU-Pro 
leaderboard_mmlu_pro
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,revision=$REVISION,dtype=auto \
    --tasks leaderboard_mmlu_pro \
    --output_path $OUTPUT_DIR \
    --batch_size 16 \
    --apply_chat_template 


# lighteval vllm "pretrained=Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "lighteval|gsm8k|5|0" \
#     --use-chat-template 

### LiveCodeBench
###lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
###    --use-chat-template \
 ##   --output-dir $OUTPUT_DIR 

uv pip install vllm==0.8.5
uv pip install flash-attn