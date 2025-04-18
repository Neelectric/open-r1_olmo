### Written by Neel Rajani, 15.04.25
### The point of this file is to let me use the Cross-Model Activation Patching techniques as proposed by Prakash et al., 2024


import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from datetime import timedelta

from utils import list_revisions



if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def run_lighteval(
    model: str,
    task: str,
    revision: str = "main",
    num_gpus: int = 1,
):
    """Evaluates a model (optionally specific revision) on a benchmark with lighteval."""
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        # push_to_hub=True,
        # hub_results_org="your user name",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
    )
    
    generation_parameters = GenerationParameters(
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        )

    model_config = VLLMModelConfig(
            pretrained=model,
            revision=revision,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            use_chat_template=True,
            data_parallel_size=num_gpus,
            max_model_length=4096,
            generation_parameters=generation_parameters,
            
    )

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        # custom_task_directory=None, # if using a custom task
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    # model = "allenai/OLMo-2-1124-7B-Instruct"
    model = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.09"
    # task = "lighteval|aime24|0|1"
    task = "lighteval|math_500|0|1"
    # revision = None
    num_gpus = 1
    
    run_lighteval(
        model=model,
        task=task,
        # revision=revision,
        num_gpus=num_gpus,
    )