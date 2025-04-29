### By Neel Rajani, 13.04.25
### Quick utility scripts

from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi
from tqdm import tqdm
import subprocess

def list_revisions(model_id: str) -> list[str]:
  """Returns all revisions of a model from the hub."""
  api = HfApi()
  refs = api.list_repo_refs(model_id)
  branch_names = [branch.name for branch in refs.branches]
  revisions = branch_names[:0:-1] 
  return revisions + ["main"]

def download_all_revisions_fast():
  # ft_model_id = "Neelectric/Qwen2.5-7B-Instruct_SFTv00.13"
  # ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv00.09"
  # ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv01.03"
  ft_model_id = "Neelectric/OLMo-2-1124-7B-Instruct_SFTv01.05"
  revisions = list_revisions(ft_model_id)
  print(revisions)
  command = [
      "HF_HUB_ENABLE_HF_TRANSFER=1",
      "huggingface-cli",
      "download",
      ft_model_id,
      "--cache-dir data/ "
      "--max-workers 32 "
      "--revision",
      "placeholder_rev"
  ]

  for revision in tqdm(revisions, dynamic_ncols=True):
    command[-1] = revision
    subprocess.run(" ".join(command), shell=True, check=True)
    
if __name__ == '__main__':
  download_all_revisions_fast()