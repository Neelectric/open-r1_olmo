from datasets import load_dataset
import glob

output_dir = "data/evals/Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10"
model_name = "Neelectric/OLMo-2-1124-7B-Instruct_GRPOv00.10"
timestamp = "latest"
task = "lighteval|gsm8k|0"

if timestamp == "latest":
    path = f"{output_dir}/results/{model_name}/*"
    timestamps = glob.glob(path)
    print(timestamps)
    timestamp = sorted(timestamps)[-1].split("/")[-1]
    print(f"Latest timestamp: {timestamp}")

details_path = f"{output_dir}/results/{model_name}/{timestamp}"

# Load the details
details = load_dataset("parquet", data_files=details_path, split="train")

for detail in details:
    print(detail)