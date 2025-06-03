uv pip uninstall transformers
uv pip uninstall vllm
uv pip uninstall flash-attn
uv pip install vllm==0.7.2

NUM_GPUS=5
NUM_TOKS=16384
MAX_GPU_USAGE=0.8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

REVISION=main
MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=auto,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=$MAX_GPU_USAGE,max_num_batched_tokens=$NUM_TOKS,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

### SimpleQA
TASK=simpleqa
# VLLM_WORKER_MULTIPROC_METHOD=spawn 
lighteval vllm $MODEL_ARGS "lighteval|simpleqa|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
