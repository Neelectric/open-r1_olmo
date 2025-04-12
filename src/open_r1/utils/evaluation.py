import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig

import os


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack thread: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]


def register_lighteval_task(
    configs: Dict[str, str], eval_suite: str, task_name: str, task_list: str, num_fewshot: int = 0
):
    """Registers a LightEval task configuration.

    - Core tasks can be added from this table: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - Custom tasks that require their own metrics / scripts, should be stored in scripts/evaluation/extended_lighteval_tasks

    Args:
        configs (Dict[str, str]): The dictionary to store the task configuration.
        eval_suite (str, optional): The evaluation suite.
        task_name (str): The name of the task.
        task_list (str): The comma-separated list of tasks in the format "extended|{task_name}|{num_fewshot}|0" or "lighteval|{task_name}|{num_fewshot}|0".
        num_fewshot (int, optional): The number of few-shot examples. Defaults to 0.
        is_custom_task (bool, optional): Whether the task is a custom task. Defaults to False.
    """
    # Format task list in lighteval format
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


LIGHTEVAL_TASKS = {}

register_lighteval_task(LIGHTEVAL_TASKS, "custom", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)


def get_lighteval_tasks():
    return list(LIGHTEVAL_TASKS.keys())


SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str, training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig"
) -> None:
    # Get task info from registered tasks
    task_list_raw = LIGHTEVAL_TASKS[benchmark]
    
    # Parse the task format - it will be in format like "custom|aime24|0|0"
    # or "extended|lcb:codegeneration|0|0"
    task_parts = task_list_raw.split("|")
    eval_suite = task_parts[0]  # 'custom', 'extended', etc.
    task = task_parts[1]  # The actual task name
    
    # Set up model configuration
    model_name = training_args.hub_model_id if hasattr(training_args, "hub_model_id") and training_args.hub_model_id else model_args.model_name_or_path
    
    # Determine appropriate number of GPUs
    num_gpus = min(8, getattr(training_args, 'num_gpus', 7))  # Default to 7 if num_gpus not specified
    
    # Create output directory
    output_dir = f"data/evals/{model_name.replace('/', '_')}"
    
    # Build model args string
    model_args_str = (
        f"pretrained={model_name},"
        f"dtype=bfloat16,"
        f"data_parallel_size={num_gpus},"
        f"max_model_length=4096,"
        f"gpu_memory_utilization=0.7,"
        f"generation_parameters={{\\\"max_new_tokens\\\":4096,\\\"temperature\\\":0.6,\\\"top_p\\\":0.95}}"
    )
    
    # Build the task format string expected by lighteval
    task_format = f"{eval_suite}|{task}|0|1"
    
    # Build command list
    cmd = [
        "lighteval", "vllm", model_args_str, task_format,
        "--custom-tasks", "src/open_r1/evaluate.py",
        "--use-chat-template",
        "--output-dir", output_dir
    ]
    
    # Add system prompt if available
    if hasattr(training_args, "system_prompt") and training_args.system_prompt:
        cmd.extend(["--system-prompt", training_args.system_prompt])
    
    print(f"Running benchmark: {benchmark}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark {benchmark}: {e}")
        # Continue with other benchmarks rather than crashing the whole process


def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
