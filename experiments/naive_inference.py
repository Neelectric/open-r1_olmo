### Written by Neel Rajani, 16.04.25
### I just wanted to look at some MLP matrices tbh

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "allenai/OLMo-2-1124-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

input_text = "Hello from planet earth"
tokenized = tokenizer(input_text, return_tensors='pt').to(model.device)
input_ids = tokenized.input_ids
output = model(input_ids)

detokenized = tokenizer.batch_decode(output)
print(detokenized)