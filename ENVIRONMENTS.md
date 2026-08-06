# Environments

## Hardware
2x 96 GB workstation GPU, 96 GB VRAM, sm_120, 188 SMs, PCIe Gen5, no NVLink
CUDA 13.2 toolkit, driver 595.84, Intel Core Ultra 9 285K, 125 GB RAM, Ubuntu 24.04
Base graphics clock: FILL_ME MHz (check: nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader)

## Clock Setup
Lock before each chapter:
  sudo nvidia-smi -i 1 -lgc FILL_ME  # GPU 1
  sudo nvidia-smi -i 0 -lgc FILL_ME  # GPU 0 (TP=2 chapters only)
Unlock: sudo nvidia-smi -rgc

## Python
Version: 3.11.x (exact: FILL_ME)
Install: pip install -e ".[dev]"

## Framework Versions (fill after install)
| Framework | Version/Commit | Install |
|---|---|---|
| vLLM | FILL_ME | pip install vllm |
| SGLang | FILL_ME | pip install sglang[all] |
| TensorRT-LLM | FILL_ME | pip install tensorrt-llm |

## Models (fill revision hash after download)
| Model | HF ID | Revision Hash |
|---|---|---|
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | FILL_ME |
| Qwen2.5-Coder-32B-Instruct | Qwen/Qwen2.5-Coder-32B-Instruct | FILL_ME |
| Llama-3.3-70B-Instruct | meta-llama/Llama-3.3-70B-Instruct | FILL_ME |

## Quantized Checkpoints
| Checkpoint | Script | Output Dir |
|---|---|---|
| Llama-3.1-8B FP8 | quantize/fp8_recipe.py | quantized/llama3-8b-fp8 |
| Llama-3.1-8B NVFP4 | quantize/nvfp4_recipe.py | quantized/llama3-8b-nvfp4 |
| Llama-3.3-70B FP8 | quantize/fp8_recipe.py | quantized/llama3-70b-fp8 |
| Llama-3.3-70B NVFP4 | quantize/nvfp4_recipe.py | quantized/llama3-70b-nvfp4 |

## Profiling Tools
Nsight Compute 2026.1, Nsight Systems 2025.6
nccl-tests: FILL_ME (https://github.com/NVIDIA/nccl-tests)
