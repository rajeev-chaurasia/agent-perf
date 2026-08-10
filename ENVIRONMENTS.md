# Environments

## Hardware

| | |
|---|---|
| GPUs | 2x 96 GB workstation GPUs (sm_120, 188 SMs) |
| Interconnect | PCIe Gen5, no NVLink |
| CPU | Intel Core Ultra 9 285K |
| RAM | 125 GB |
| OS | Ubuntu 24.04 |
| CUDA toolkit | 13.2 |
| Driver | 595.84 |
| Base graphics clock | 3090 MHz (both GPUs, verified via `nvidia-smi --query-gpu=clocks.max.gr`) |

GPU 0 is the display adapter. GPU 1 is the primary benchmark GPU. Chapters 1-3 use GPU 1 only. Chapter 4 uses both (TP=2).

## Clock setup

Lock clocks before running any chapter:

```bash
sudo nvidia-smi -i 1 -lgc 3090   # GPU 1 (benchmark GPU)
sudo nvidia-smi -i 0 -lgc 3090   # GPU 0 (TP=2 chapters only)
```

Release after:

```bash
sudo nvidia-smi -rgc
```

The run scripts handle lock/release automatically via a `trap cleanup EXIT` handler.

## Python

Version: 3.12.3

Base install (includes the harness CLI tools):

```bash
pip install -e ".[dev]"
```

## Framework versions

Each framework runs in its own venv to avoid torch version conflicts.

| Framework | Version | Venv | torch |
|---|---|---|---|
| vLLM | 0.26.0 | `.venv-vllm` | 2.11.0+cu130 |
| SGLang | 0.5.9 | `.venv-sglang` | 2.9.1+cu126 |
| TensorRT-LLM | 1.3.0rc23 | `.venv-trtllm` | 2.11.0+cu130 |

TRT-LLM 1.3.0rc23 uses the PyTorch backend rather than compiled TRT engines. Model loading takes ~26s and allocates ~65 GB KV cache on a 96 GB GPU. The `ninja` build tool is required for FlashInfer JIT compilation and must be installed inside `.venv-trtllm`:

```bash
uv pip install ninja --python .venv-trtllm/bin/python
```

## Models

| Model | HF ID | Revision |
|---|---|---|
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Qwen2.5-Coder-32B-Instruct | Qwen/Qwen2.5-Coder-32B-Instruct | 381fc969f78efac66bc87ff7ddeadb7e73c218a7 |
| Llama-3.3-70B-Instruct | meta-llama/Llama-3.3-70B-Instruct | 6f6073b423013f6a7d4d9f39144961bfbfbc386b |

Download to `models/`:

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/llama-3.1-8b-instruct
```

## Quantized checkpoints

| Checkpoint | Script | Output |
|---|---|---|
| Llama-3.1-8B FP8 | `quantize/fp8_recipe.py` | `quantized/llama3-8b-fp8` |
| Llama-3.1-8B NVFP4 | `quantize/nvfp4_recipe.py` | `quantized/llama3-8b-nvfp4` |
| Llama-3.3-70B FP8 | `quantize/fp8_recipe.py` | `quantized/llama3-70b-fp8` |
| Llama-3.3-70B NVFP4 | `quantize/nvfp4_recipe.py` | `quantized/llama3-70b-nvfp4` |

## VRAM fit (BF16, single GPU)

| Model | BF16 size | Fits in 96 GB | Notes |
|---|---|---|---|
| Llama-3.1-8B-Instruct | ~16 GB | Yes | All precisions, Ch1-Ch3 |
| Qwen2.5-Coder-32B-Instruct | ~65 GB | Yes | All precisions, Ch1-Ch3 |
| Llama-3.3-70B-Instruct | ~141 GB | No (BF16) | FP8 ~70 GB, NVFP4 ~35 GB; Ch2+ |

## sm_120 notes

- Flash Attention 2 is not compiled for sm_120; FlashInfer or FA3 is required.
- NVFP4 KV-cache block scale writes: verify fix from vLLM PR #50085 is present.
- FlashInfer sampler JIT failures on first request: install pre-compiled wheels or use `ninja` inside the venv.
- Requires CUDA 12.8+ and PyTorch 2.6+ for full sm_120 support.

## Profiling tools

- Nsight Compute 2026.1
- Nsight Systems 2025.6
- nccl-tests: https://github.com/NVIDIA/nccl-tests
