### Written by Neel Rajani, 15.04.25
### The point of this file is to let me use the Cross-Model Activation Patching techniques as proposed by Prakash et al., 2024


import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig

from lighteval.models.transformers.transformers_model import TransformersModelConfig

from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
# from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from lighteval.metrics.metrics import Metrics
from datetime import timedelta

from utils import list_revisions
import wandb
import tqdm

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
    max_model_len: int = 4096,
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
    )
    
    generation_parameters = GenerationParameters(
        max_new_tokens=max_model_len,
        temperature=0.6,
        top_p=0.95,
        )

    model_config = VLLMModelConfig(
            model_name=model,
            revision=revision,
            gpu_memory_utilization=0.9,
            dtype="auto",
            use_chat_template=True,
            data_parallel_size=num_gpus,
            max_model_length=max_model_len,
            generation_parameters=generation_parameters,
    )
    
    # model_config = TransformersModelConfig(
    #     pretrained=model,
    #     revision=revision,
    #     accelerator=accelerator,
    #     # device="cuda",
    #     batch_size="16",
    #     dtype="auto",
    #     use_chat_template=True,
    #     # model_parallel=True,
    #     generation_parameters=generation_parameters,
    #     max_length=4096,
    #     max_gen_toks=4096,
    # )

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        # custom_task_directory=None, # if using a custom task
        metric_options={
            "gpqa_pass@1:1_samples": {"num_samples": 1},
        }

    )
    
    
    pipeline.evaluate()
    result = pipeline.get_results()
    # print(result)
    # print(f"NOW PRINTING result[results]")
    # print(result["results"])
    save_result = pipeline.save_and_push_results()
    show_result = pipeline.show_results()
    
    return result["results"]


if __name__ == "__main__":
    model_id = "EleutherAI/pythia-14m"
    max_model_len = 2048
    # model = "allenai/OLMo-2-1124-7B-Instruct"
    # model = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.09"
    # model = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10"
    # task = "lighteval|aime24|0|1"
    task = "lighteval|gpqa:diamond|0|0"
    # revision = None
    num_gpus = 8
    
    # revisions = list_revisions(model_id=model_id)
    # print(f"Found {len(revisions)} revisions for {model_id}: {revisions}")
    # assert len(revisions) == 20   
    # for revision in tqdm(revisions, desc=f"Processing {model_id}"):
    #     pass
    
    result = run_lighteval(
        model=model_id,
        task=task,
        # revision=revision,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
    )
    print(f"top level function gets {result}")
    # api = wandb.Api()

    # run = api.run("<entity>/<project>/<run_id>")
    # run.config["key"] = updated_value
    # run.update()