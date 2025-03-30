wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv
source .venv/bin/activate
make install

uv pip install liger-kernel

uv pip install huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download open-r1/OpenR1-Math-220k --repo-type dataset
huggingface-cli download allenai/OLMo-2-1124-7B-Instruct
apt install gettext
pip install gpustat