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
from tqdm import tqdm
from pathlib import Path
import json

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
            gpu_memory_utilization=0.8,
            dtype="auto",
            use_chat_template=True,
            data_parallel_size=num_gpus,
            max_model_length=max_model_len,
            max_num_batched_tokens=max_model_len,
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
        # metric_options={
        #     "gpqa_pass@1:1_samples": {"num_samples": 1},
        #     "gpqa_pass@1:4_samples": {"num_samples": 1},
        #     "gpqa_pass@1:8_samples": {"num_samples": 1}
        # }
    )
    pipeline.evaluate()
    result = pipeline.get_results()

    # save_result = pipeline.save_and_push_results()
    show_result = pipeline.show_results()
    
    return result["results"]


def perform_eval(ft_model_id, 
                 max_model_len, 
                 task, 
                 task_entry, 
                 task_entry_result, 
                 task_filename, 
                 num_gpus):
    
    save_path = "results/python_evals/" + task_filename + "/" + ft_model_id
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path + f"{task_filename}.json", "r") as f:
            results_dict = json.load(f)
        print(f"Found and loaded results_dict for {ft_model_id} on {task}!")
    except:
        print(f"Did not find results_dict for {ft_model_id} on {task}, under {save_path}, computing it now...") 
        results_dict = {}
    
    
    revisions = list_revisions(model_id=ft_model_id)
    print(f"Found {len(revisions)} revisions for {ft_model_id}: {revisions}")
    # assert len(revisions) == 20   
    for revision in tqdm(revisions, desc=f"Processing {ft_model_id}", dynamic_ncols=True):
        result = run_lighteval(
            model=ft_model_id,
            task=task,
            revision=revision,
            num_gpus=num_gpus,
            max_model_len=max_model_len,
        )
        print(f"top level function gets {result}")
        result_pass_at1_1 = result[task_entry][task_entry_result]
        print("*"*200)
        print(f"result at 1 for revision {revision} seems to be {result_pass_at1_1}")
        print("*"*200)
        
        # save results and store in json
        results_dict[revision] = result_pass_at1_1
        with open(save_path + f"/{task_filename}.json", "w") as f:
            json.dump(results_dict, f)
    print(f"final results:\n{results_dict}")
    return


if __name__ == "__main__":
    # model = "allenai/OLMo-2-1124-7B-Instruct"
    ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv02.08"
    # ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.14"
    max_model_len = 4096
    num_gpus = 7
    
    # # aime24
    # task = "lighteval|aime24|0|1"
    # task_entry = "lighteval:aime24:0"
    # task_entry_result = "math_pass@1:1_samples"
    # task_filename = task.split("|")[1]
    # perform_eval(ft_model_id=ft_model_id, max_model_len=max_model_len, task=task, task_entry=task_entry, task_entry_result=task_entry_result, task_filename=task_filename, num_gpus=num_gpus)
    
    # math_500
    task = "lighteval|math_500|0|0"
    task_entry = "lighteval:math_500:0"
    task_entry_result = "math_pass@1:1_samples"
    task_filename = task.split("|")[1]
    perform_eval(ft_model_id=ft_model_id, max_model_len=max_model_len, task=task, task_entry=task_entry, task_entry_result=task_entry_result, task_filename=task_filename, num_gpus=num_gpus)
    
    # # gpqa diamond
    # task = "lighteval|gpqa:diamond|0|0"
    # task_entry = "lighteval:gpqa:diamond:0"
    # task_entry_result = "gpqa_pass@1:1_samples"
    # perform_eval(ft_model_id=ft_model_id, max_model_len=max_model_len, task=task, task_entry=task_entry, task_entry_result=task_entry_result, task_filename=task_filename, num_gpus=num_gpus)