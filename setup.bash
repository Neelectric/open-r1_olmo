wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv
source .venv/bin/activate
make install

uv pip install liger-kernel

uv pip install huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1

uv pip install fire
uv pip install matplotlib
uv pip install seaborn
uv pip install imageio
uv pip install flashinfer-python

uv pip install gpustat

uv pip install -U trl
uv pip uninstall flash-attn
uv pip install -U flash-attn
uv pip install -U transformers
uv pip install -U vllm

# uv pip install bigbench

huggingface-cli download open-r1/OpenR1-Math-220k --repo-type dataset
apt install gettext -y
sudo apt-get install slurm-wlm

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
uv pip install -e ".[math,ifeval,sentencepiece]"
