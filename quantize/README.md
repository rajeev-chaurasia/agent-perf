# Quantization Recipes

## FP8 vs NVFP4

| | FP8 | NVFP4 |
|---|---|---|
| Scheme | W8A8 static | Block-FP4 weights |
| Target hardware | sm_89+ | sm_120 |
| Serving stack | vLLM (native support) | vLLM 0.26+ (--quantization nvfp4) |
| Compression ratio | ~2x vs BF16 | ~4x vs BF16 |
| Quality trade-off | Good throughput/quality balance | Higher compression; minor accuracy drop |

**FP8**: Uses static activation calibration (stats baked into checkpoint at quantisation time) — no per-token runtime overhead. Widely supported by vLLM without additional conversion steps.

**NVFP4**: block-wise FP4 weight format native to sm_120. Served via vLLM 0.26+ using `--quantization nvfp4`. Yields the highest compression ratio of the two formats.

## Prerequisites

### FP8

```bash
pip install llm-compressor
```

### NVFP4

```bash
pip install "nvidia-modelopt[torch]"
```

## Example Commands

### FP8

```bash
python quantize/fp8_recipe.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir quantized/llama3-8b-fp8
```

### NVFP4

```bash
python quantize/nvfp4_recipe.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir quantized/llama3-8b-nvfp4 \
  --device cuda:1
```

## Output

Each script writes a `recipe.json` alongside the checkpoint:

```json
// FP8
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "scheme": "FP8_W8A8_static",
  "calib_dataset": "open-platypus",
  "calib_nsamples": 512,
  "duration_s": 312.4
}

// NVFP4
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "scheme": "NVFP4",
  "calib_size": 512,
  "device": "cuda:1",
  "duration_s": 408.1
}
```

## CLI Reference

### fp8_recipe.py

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model id or local path |
| `--output-dir` | *(required)* | Destination directory |
| `--calib-dataset` | `open-platypus` | Calibration dataset name |
| `--calib-nsamples` | `512` | Number of calibration samples |
| `--max-seq-len` | `2048` | Max sequence length during calibration |
| `--force` | `false` | Overwrite existing output directory |

### nvfp4_recipe.py

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model id or local path |
| `--output-dir` | *(required)* | Destination directory |
| `--calib-size` | `512` | Calibration dataset size |
| `--device` | `cuda:1` | Torch device for quantisation |
| `--force` | `false` | Overwrite existing output directory |
