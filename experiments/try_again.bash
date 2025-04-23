NUM_GPUS=8
NUM_TOKS=4096
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.12
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,pipeline_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,max_num_batched_tokens=$NUM_TOKS,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR