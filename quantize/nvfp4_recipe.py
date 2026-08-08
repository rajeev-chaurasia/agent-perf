"""NVFP4 quantization recipe using TensorRT Model Optimizer (modelopt).

Produces a quantised checkpoint targeting sm_120 (sm_120) GPU architecture
for serving with vLLM 0.26+ on sm_120 (sm_120).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def quantize(
    model_id: str,
    output_dir: str,
    calib_size: int,
    device: str,
) -> None:
    # Lazy import: nvidia-modelopt is a large GPU-bound optional dependency;
    # import at call time so the module is importable on CPU-only machines for
    # --help and other non-quantisation uses.
    import modelopt.torch.quantize as mtq  # type: ignore[import-untyped]

    out = Path(output_dir)
    if out.exists():
        # Refuse silently overwriting completed runs to preserve reproducibility.
        print(
            f"[nvfp4_recipe] Output directory already exists: {out}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[nvfp4_recipe] Quantising {model_id} -> {out}  (NVFP4, device={device})")

    # FP4 requires block-wise weight scaling; modelopt handles calibration
    # statistics collection internally when given a calib_size.
    quant_cfg = mtq.FP4_DEFAULT_CFG  # W4A16 block-fp4 targeting sm_120

    t0 = time.monotonic()
    mtq.quantize(
        model=model_id,
        quant_cfg=quant_cfg,
        calib_size=calib_size,
        device=device,
        output_dir=str(out),
    )
    duration_s = time.monotonic() - t0

    meta = {
        "model": model_id,
        "scheme": "NVFP4",
        "calib_size": calib_size,
        "device": device,
        "duration_s": round(duration_s, 2),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.json").write_text(json.dumps(meta, indent=2))
    print(f"[nvfp4_recipe] Done in {duration_s:.1f}s. Recipe written to {out / 'recipe.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantise a HuggingFace model to NVFP4 via TensorRT Model Optimizer."
    )
    parser.add_argument("--model", required=True, dest="model_id", help="HuggingFace model id or local path")
    parser.add_argument("--output-dir", required=True, help="Directory for the quantised checkpoint")
    parser.add_argument("--calib-size", type=int, default=512, help="Calibration dataset size (default: 512)")
    parser.add_argument("--device", default="cuda:1", help="Torch device for quantisation (default: cuda:1)")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists")

    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists() and not args.force:
        print(
            f"[nvfp4_recipe] Output directory already exists: {out}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    if out.exists() and args.force:
        import shutil
        shutil.rmtree(out)

    quantize(
        model_id=args.model_id,
        output_dir=args.output_dir,
        calib_size=args.calib_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
