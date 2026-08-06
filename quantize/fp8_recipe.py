"""FP8 (W8A8 static) quantization recipe using llm-compressor.

Produces a HuggingFace-compatible checkpoint ready for vLLM serving.
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
    calib_dataset: str,
    calib_nsamples: int,
    max_seq_len: int,
) -> None:
    # Lazy imports: llm-compressor is a large optional dependency; importing at
    # call time avoids penalising scripts that only need other parts of the package.
    from llmcompressor import oneshot  # type: ignore[import-untyped]
    from llmcompressor.modifiers.quantization import QuantizationModifier  # type: ignore[import-untyped]

    out = Path(output_dir)
    if out.exists():
        # Caller must pass --force to overwrite an existing checkpoint; otherwise
        # we risk silently corrupting a completed quantization run.
        print(
            f"[fp8_recipe] Output directory already exists: {out}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[fp8_recipe] Quantising {model_id} -> {out}  (W8A8 FP8 static)")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_W8A8",
        # Static activation calibration avoids per-token runtime overhead;
        # calibration stats are collected once and baked into the checkpoint.
        ignore=["lm_head"],
    )

    t0 = time.monotonic()
    oneshot(
        model=model_id,
        recipe=recipe,
        dataset=calib_dataset,
        num_calibration_samples=calib_nsamples,
        max_seq_length=max_seq_len,
        output_dir=str(out),
        save_compressed=True,
    )
    duration_s = time.monotonic() - t0

    meta = {
        "model": model_id,
        "scheme": "FP8_W8A8_static",
        "calib_dataset": calib_dataset,
        "calib_nsamples": calib_nsamples,
        "duration_s": round(duration_s, 2),
    }
    (out / "recipe.json").write_text(json.dumps(meta, indent=2))
    print(f"[fp8_recipe] Done in {duration_s:.1f}s. Recipe written to {out / 'recipe.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantise a HuggingFace model to FP8 W8A8 static via llm-compressor."
    )
    parser.add_argument("--model", required=True, dest="model_id", help="HuggingFace model id or local path")
    parser.add_argument("--output-dir", required=True, help="Directory for the quantised checkpoint")
    parser.add_argument("--calib-dataset", default="open-platypus", help="Dataset name for calibration (default: open-platypus)")
    parser.add_argument("--calib-nsamples", type=int, default=512, help="Number of calibration samples (default: 512)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length during calibration (default: 2048)")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists")

    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists() and not args.force:
        print(
            f"[fp8_recipe] Output directory already exists: {out}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Remove the directory guard inside quantize() by deleting it before calling
    # when --force is set; the inner check fires only when the directory persists.
    if out.exists() and args.force:
        import shutil
        shutil.rmtree(out)

    quantize(
        model_id=args.model_id,
        output_dir=args.output_dir,
        calib_dataset=args.calib_dataset,
        calib_nsamples=args.calib_nsamples,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
