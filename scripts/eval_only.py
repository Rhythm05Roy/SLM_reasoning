#!/usr/bin/env python3
"""
scripts/eval_only.py — Standalone Evaluation Script
====================================================
Load an already-trained adapter and run the full evaluation suite
without re-training.  Useful for:
  • Evaluating a checkpoint from a resumed session
  • Running ablations on a published adapter
  • Generating paper-ready plots from saved results

Usage
-----
  python scripts/eval_only.py \
    --adapter_path /workspace/curricsym_checkpoints/grpo_final \
    --output_dir   /workspace/curricsym_output \
    --max_eval_examples 400
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from curricsym.configs import TrainingConfig
from curricsym.utils import get_logger, set_seed, vram_summary
from curricsym.data import build_all_datasets
from curricsym.models import load_model_and_tokenizer, SymbolicVerifier
from curricsym.training import CurriculumScheduler
from curricsym.evaluation import run_full_evaluation, generate_full_dashboard
from curricsym.reporting import build_experiment_report

logger = get_logger("eval_only", logging.INFO)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CurricSym — standalone evaluation")
    p.add_argument("--adapter_path", required=True,
                   help="Path to a saved adapter dir (grpo_final or specific stage)")
    p.add_argument("--model_name", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--output_dir", default="/workspace/curricsym_eval_output")
    p.add_argument("--max_eval_examples", type=int, default=400)
    p.add_argument("--internalization_eval_examples", type=int, default=300)
    p.add_argument("--gsm_symbolic_size", type=int, default=3000)
    p.add_argument("--proofwriter_size", type=int, default=2000)
    p.add_argument("--stage_metrics_json", default="",
                   help="Path to grpo_stage_metrics.json (for ablation plots)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CurricSym-SLM-Lite — Standalone Evaluation")
    logger.info("=" * 60)
    logger.info(f"Adapter : {args.adapter_path}")
    logger.info(f"Output  : {args.output_dir}")
    logger.info(f"VRAM    :\n{vram_summary()}")

    # Build a minimal config for evaluation
    config = TrainingConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir + "/ckpt",
        data_cache_dir=args.output_dir + "/data_cache",
        max_eval_examples=args.max_eval_examples,
        internalization_eval_examples=args.internalization_eval_examples,
        gsm_symbolic_size=args.gsm_symbolic_size,
        proofwriter_size=args.proofwriter_size,
        use_wandb=False,
        seed=args.seed,
    )

    # Load data
    datasets = build_all_datasets(config)
    eval_verified = datasets["eval_verified"]
    paired_ablation = datasets["paired_ablation"]

    # Load model + adapter
    model, tokenizer = load_model_and_tokenizer(config)
    from curricsym.models import load_sft_adapter
    load_sft_adapter(model, args.adapter_path)

    # Verifier + dummy curriculum (for reporting)
    verifier = SymbolicVerifier()
    curriculum = CurriculumScheduler(n_stages=3)
    curriculum.set_difficulty_range(0.0, 5.0)

    # Load stage metrics if provided
    stage_metrics = []
    if args.stage_metrics_json and Path(args.stage_metrics_json).exists():
        with open(args.stage_metrics_json) as f:
            stage_metrics = json.load(f)
        logger.info(f"Loaded stage metrics: {args.stage_metrics_json}")

    # Run full evaluation
    results = run_full_evaluation(
        config=config,
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        eval_verified=eval_verified,
        paired_ablation=paired_ablation,
        stage_metrics=stage_metrics,
        curriculum=curriculum,
    )

    # Build report
    if stage_metrics:
        build_experiment_report(
            config=config,
            model=model,
            tokenizer=tokenizer,
            results=results,
            stage_metrics=stage_metrics,
            curriculum=curriculum,
            verifier=verifier,
        )
        generate_full_dashboard(
            stage_metrics=stage_metrics,
            with_tools=results.get("with_tools", {}),
            without_tools=results.get("without_tools", {}),
            internalization_results=results.get("internalization", {}),
            consistency_rate=results.get("consistency_rate", 0.0),
            output_dir=args.output_dir,
        )

    logger.info("✅ Evaluation complete")


if __name__ == "__main__":
    main()
