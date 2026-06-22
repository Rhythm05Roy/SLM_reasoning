#!/usr/bin/env python3
"""
scripts/push_to_hub.py — Push saved adapter/merged model to HuggingFace Hub
============================================================================
Usage
-----
  HF_TOKEN=hf_xxx python scripts/push_to_hub.py \
    --adapter_path /workspace/curricsym_checkpoints/grpo_final \
    --repo_id      your-username/curricsym-qwen2.5-3b \
    --push_adapter          # push LoRA adapter only (small)
    --push_merged           # push merged 16-bit model (large)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from curricsym.configs import TrainingConfig
from curricsym.models import load_model_and_tokenizer, load_sft_adapter
from curricsym.utils import get_logger

logger = get_logger("push_to_hub")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--repo_id", required=True, help="HF repo, e.g. user/model-name")
    p.add_argument("--model_name", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--push_adapter", action="store_true",
                   help="Push LoRA adapter only (~100 MB)")
    p.add_argument("--push_merged", action="store_true",
                   help="Push merged BF16 model (~6 GB)")
    p.add_argument("--quantization", default="bf16",
                   choices=["bf16", "f16", "q4_k_m"],
                   help="Quantization for merged model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        sys.exit(1)

    config = TrainingConfig(model_name=args.model_name)
    model, tokenizer = load_model_and_tokenizer(config)
    load_sft_adapter(model, args.adapter_path)

    if args.push_adapter:
        logger.info(f"Pushing adapter → {args.repo_id}-adapter")
        model.push_to_hub(f"{args.repo_id}-adapter", token=hf_token)
        tokenizer.push_to_hub(f"{args.repo_id}-tokenizer", token=hf_token)
        logger.info("Adapter pushed ✅")

    if args.push_merged:
        logger.info(f"Pushing merged {args.quantization} model → {args.repo_id}")
        model.push_to_hub_merged(
            args.repo_id,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        logger.info("Merged model pushed ✅")


if __name__ == "__main__":
    main()
