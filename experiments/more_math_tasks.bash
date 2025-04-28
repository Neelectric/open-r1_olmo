NUM_GPUS=8
NUM_TOKS=4096
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.04
# MODEL=allenai/OLMo-2-1124-7B-Instruct
# MODEL=Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL


# # ### gaokao
# CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS "lighteval|agieval:gaokao-mathqa|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# ### bigbench|mathematical_induction
# CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS "bigbench|mathematical_induction|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# ### lighteval|agieval:sat-math
# CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS "lighteval|agieval:sat-math|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

#### lighteval|math
# CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS "lighteval|mathqa|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

## hendrycks_math
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=auto, \
    --tasks hendrycks_math \
    --num_fewshot 4 \
    --output_path $OUTPUT_DIR \
    --batch_size 64 \
    --apply_chat_template 
