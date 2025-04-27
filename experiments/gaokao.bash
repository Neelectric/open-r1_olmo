NUM_GPUS=8
NUM_TOKS=4096
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.04
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL


# ### gaokao
CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS "lighteval|agieval:gaokao-mathqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
