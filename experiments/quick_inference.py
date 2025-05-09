from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/OLMo-2-1124-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    torch_dtype = "auto"
)
tok = AutoTokenizer.from_pretrained(model_id)


print(model)
print(model.model.layers[0].mlp.gate_proj.weight)

prompt ="hello"
inputs = tok(prompt, return_tensors="pt").to(model.device)
output = model(**inputs)
print(output)