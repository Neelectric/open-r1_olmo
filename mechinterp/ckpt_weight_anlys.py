### By Neel Rajani, 13.04.25
### Here we do analysis of a model's weights across training

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from tqdm import tqdm
import torch
import fire



### ok what is it we want in here
# i want to give an hf model name
# and get an analysis of the weights changing across training as a result
# so probably a 'figures' folder right
# into which we paste graphs which plot changes across steps



def hello(name):
  return f'Hello {name}!'

        
        
def list_revisions(model_id: str) -> list[str]:
  """Returns all revisions of a model from the hub."""
  api = HfApi()
  refs = api.list_repo_refs(model_id)
  branch_names = [branch.name for branch in refs.branches]
  revisions = branch_names[:0:-1] 
  return revisions

def compare_base_and_ckpt(base_model_id: str, ft_model_id: str, checkpoint_id: str) -> dict:
  """Compares the weights of a base model and a given checkpoint, using the normalized Frobenius norm of differences and standard deviations."""
  base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    attn_implementation="flash_attention_2",
    device_map="cpu",
  )
  ckpt_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    attn_implementation="flash_attention_2",

    device_map="cpu",
  )
  # sanity check 
  base_params = dict(base_model.named_parameters())
  ckpt_params = dict(ckpt_model.named_parameters())
  assert base_params.keys() == ckpt_params.keys()
  
  results = {}
  for name, base_param in base_model.named_parameters():
    ckpt_param = ckpt_params[name]
    norm_of_diff = torch.linalg.norm(ckpt_param - base_param).item()
    norm_of_base = torch.linalg.norm(base_param).item()
    rel_diff = norm_of_diff / norm_of_base if norm_of_base > 0 else float('inf')
    results[name] = {
      "frob_norm_of_diff": norm_of_diff,
      "forb_norm_of_rel_diff": rel_diff,
      'param_size': base_param.numel()
    }
  return results


def main():
  base_model_id = "Qwen/Qwen2.5-7B-Instruct"
  ft_model_id = "Neelectric/Qwen2.5-7B-Instruct_SFTv00.13"
  revisions = list_revisions(ft_model_id)
  print(revisions)
  
  compare_base_and_ckpt(base_model_id, ft_model_id, revisions[0])
  
  
  
  
  
  # fire.Fire(hello)

if __name__ == '__main__':
  main()