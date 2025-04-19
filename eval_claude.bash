#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
set -x -e

source ~/.bashrc
source openr1/bin/activate

# Define model and configuration
MODEL=Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10
MODEL_REVISION="main"
NUM_GPUS=8
MAX_TOKENS=4096
MAX_GPU_MEM_USAGE=0.8

# Extract model name for repository naming
MODEL_NAME=$(echo $MODEL | sed 's/\//_/g') # replaces / with _

# Set repository IDs and output directory
LM_EVAL_REPO_ID="open-r1/open-r1-eval-leaderboard"
DETAILS_REPO_ID="open-r1/details-$MODEL_NAME"
OUTPUT_DIR="data/evals/$MODEL"

# Disable DeepSpeed to avoid conflicts
ACCELERATE_USE_DEEPSPEED=false

# Enable fast downloads
HF_HUB_ENABLE_HF_TRANSFER=1

# We'll use tensor parallelism for this model
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_ARGS="pretrained=$MODEL,revision=$MODEL_REVISION,trust_remote_code=False,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_TOKENS,max_num_batched_tokens=$MAX_TOKENS,gpu_memory_utilization=$MAX_GPU_MEM_USAGE,generation_parameters={max_new_tokens:$MAX_TOKENS,temperature:0.6,top_p:0.95}"

# Function to run evaluation for a single task
run_evaluation() {
    local TASK_NAME=$1
    local TASK_CONFIG=$2
    
    echo "Running evaluation for $TASK_NAME..."
    
    # Create task-specific output directory
    local TASK_OUTPUT_DIR="$OUTPUT_DIR/$TASK_NAME"
    mkdir -p "$TASK_OUTPUT_DIR"
    
    # Run lighteval
    lighteval vllm "$MODEL_ARGS" "$TASK_CONFIG" \
        --use-chat-template \
        --output-dir "$TASK_OUTPUT_DIR"
    
    # Upload results to HF hub
    echo "Uploading results for $TASK_NAME to Hugging Face Hub..."
    OUTPUT_FILEPATHS=$(find "$TASK_OUTPUT_DIR/results/" -type f \( -name "*.json" \))
    for filepath in $OUTPUT_FILEPATHS; do
        echo "Uploading $filepath to Hugging Face Hub..."
        filename=$(basename -- "$filepath")
        for attempt in {1..20}; do
            if huggingface-cli upload --repo-type space --private $LM_EVAL_REPO_ID $filepath "$TASK_OUTPUT_DIR/$filename"; then
                echo "Upload succeeded for $filepath"
                break
            else
                echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in 5 seconds..."
                sleep 5
            fi
        done
    done
    
    # Upload details to HF hub
    echo "Uploading details for $TASK_NAME to Hugging Face Hub..."
    DETAILS_FILEPATHS=$(find "$TASK_OUTPUT_DIR/details/" -type f \( -name "*.parquet" \))
    echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
    TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
    python scripts/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --config_name $MODEL_REVISION.$TASK_NAME.$TIMESTAMP
}

# Run evaluations for each task
echo "Starting evaluations for $MODEL"

# AIME 2024
run_evaluation "aime24" "lighteval|aime24|0|0"

# MATH-500
run_evaluation "math_500" "lighteval|math_500|0|0"

# GPQA Diamond
run_evaluation "gpqa_diamond" "lighteval|gpqa:diamond|0|0"

# ifeval
run_evaluation "ifeval" "extended|ifeval|0|0"

# GSM8k
run_evaluation "gsm8k" "lighteval|gsm8k|5|0"

# MMLU using lighteval
run_evaluation "mmlu" "leaderboard|mmlu|0|0"

# MMLU-Pro using lm-eval harness (special handling)
echo "Running MMLU-Pro evaluation..."
MMLU_PRO_OUTPUT_DIR="$OUTPUT_DIR/mmlu_pro"
mkdir -p "$MMLU_PRO_OUTPUT_DIR"

accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=auto," \
    --tasks leaderboard_mmlu_pro \
    --output_path "$MMLU_PRO_OUTPUT_DIR" \
    --batch_size 16 \
    --apply_chat_template

echo "All evaluations completed!"
echo "Cleaning up..."

echo "Done!"