#!/usr/bin/env python3
"""
train.py — CurricSym-SLM-Lite Main Training Script
====================================================
Orchestrates the full pipeline:

  Phase 0: Environment setup & data loading
  Phase 1: SFT warm-up
  Phase 2: Staged GRPO RL (3 curriculum stages)
  Phase 3: Evaluation (accuracy, ablations, internalization)
  Phase 4: Reporting (JSON report + plots)
  Phase 5: (optional) Push to HuggingFace Hub

Usage on RunPod RTX 4090
-------------------------
  # Basic run
  python train.py

  # Override paths via env vars
  OUTPUT_DIR=/workspace/out CHECKPOINT_DIR=/workspace/ckpt python train.py

  # Resume after SFT (skip SFT phase)
  python train.py --resume_from_sft /workspace/ckpt/sft_adapter

  # Resume from GRPO stage 1
  python train.py --resume_from_sft /workspace/ckpt/sft_adapter \
                  --resume_grpo_stage 1

  # Push to HuggingFace Hub after training
  python train.py --push_to_hub --hf_repo your-username/curricsym-qwen2.5-3b

  # Wandb disabled (for quick test)
  python train.py --no_wandb --grpo_max_steps 30 --sft_max_steps 50

CLI flags mirror TrainingConfig fields so you can tune without editing code.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16'
import sys
from pathlib import Path

# ── Make sure the package is importable when running as a script ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curricsym.configs import TrainingConfig
from curricsym.utils import (
    get_logger,
    set_seed,
    vram_summary,
    log_environment,
    list_checkpoints,
)
from curricsym.data import build_all_datasets
from curricsym.models import (
    load_model_and_tokenizer,
    SymbolicVerifier,
    save_adapter,
    merge_and_save_full_model,
)
from curricsym.training import (
    run_sft,
    run_grpo,
    CurriculumScheduler,
)
from curricsym.evaluation import (
    run_full_evaluation,
    generate_full_dashboard,
    plot_stage_losses,
    plot_tool_fading,
    plot_accuracy_comparison,
    plot_process_quality,
    plot_internalization_delta,
)
from curricsym.reporting import build_experiment_report

logger = get_logger("train", logging.INFO)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CurricSym-SLM-Lite — RTX 4090 Training Script"
    )

    # Model
    p.add_argument("--model_name", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use QLoRA 4-bit (for 7B+ models or <24 GB GPU)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)

    # SFT
    p.add_argument("--sft_lr", type=float, default=2e-4)
    p.add_argument("--sft_batch_size", type=int, default=4)
    p.add_argument("--sft_max_steps", type=int, default=500)

    # GRPO
    p.add_argument("--grpo_lr", type=float, default=5e-6)
    p.add_argument("--grpo_batch_size", type=int, default=8)
    p.add_argument("--grpo_num_generations", type=int, default=8)
    p.add_argument("--grpo_max_steps", type=int, default=300)

    # Data
    p.add_argument("--gsm_symbolic_size", type=int, default=3000)
    p.add_argument("--proofwriter_size", type=int, default=2000)

    # Paths
    p.add_argument("--output_dir", default="")
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--data_cache_dir", default="")

    # Resume
    p.add_argument("--resume_from_sft", default="",
                   help="Path to SFT adapter dir — skips SFT phase")
    p.add_argument("--resume_grpo_stage", type=int, default=0,
                   help="GRPO stage to resume from (0 = start fresh)")

    # W&B
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb_project", default="curricsym-slm-lite")

    # Eval
    p.add_argument("--max_eval_examples", type=int, default=300)
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip evaluation phases (useful for quick smoke tests)")

    # Export
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push merged model to HuggingFace Hub after training")
    p.add_argument("--hf_repo", default="",
                   help="HuggingFace repo ID (e.g. user/model-name)")
    p.add_argument("--merge_model", action="store_true",
                   help="Merge LoRA adapters into full BF16 model and save")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = TrainingConfig()

    # Override with CLI args
    cfg.model_name = args.model_name
    cfg.max_seq_length = args.max_seq_length
    cfg.load_in_4bit = args.load_in_4bit
    cfg.lora_r = args.lora_r
    cfg.lora_alpha = args.lora_alpha

    cfg.sft_lr = args.sft_lr
    cfg.sft_batch_size = args.sft_batch_size
    cfg.sft_max_steps = args.sft_max_steps

    cfg.grpo_lr = args.grpo_lr
    cfg.grpo_batch_size = args.grpo_batch_size
    cfg.grpo_num_generations = args.grpo_num_generations
    cfg.grpo_max_steps = args.grpo_max_steps

    cfg.gsm_symbolic_size = args.gsm_symbolic_size
    cfg.proofwriter_size = args.proofwriter_size

    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.data_cache_dir:
        cfg.data_cache_dir = args.data_cache_dir

    cfg.resume_from_sft = args.resume_from_sft
    cfg.resume_from_grpo_stage = args.resume_grpo_stage
    cfg.use_wandb = not args.no_wandb
    cfg.wandb_project = args.wandb_project
    cfg.max_eval_examples = args.max_eval_examples
    cfg.seed = args.seed

    # Trigger __post_init__ side-effects (dir creation, run name)
    cfg.__post_init__()
    return cfg


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
def _setup_wandb(config: TrainingConfig) -> None:
    if not config.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return
    try:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
            resume="allow",
        )
        logger.info(f"W&B initialised — project={config.wandb_project}")
    except ImportError:
        logger.warning("wandb not installed — disabling W&B logging")
        config.use_wandb = False
        os.environ["WANDB_DISABLED"] = "true"


def _generate_plots(
    config,
    stage_metrics,
    with_tools,
    without_tools,
    internalization_results,
    consistency_rate,
) -> None:
    logger.info("Generating result figures...")
    plot_stage_losses(stage_metrics, config.output_dir)
    plot_tool_fading(stage_metrics, config.output_dir)
    plot_accuracy_comparison(with_tools, without_tools, config.output_dir)
    plot_process_quality(with_tools, consistency_rate, config.output_dir)
    plot_internalization_delta(internalization_results, config.output_dir)
    generate_full_dashboard(
        stage_metrics=stage_metrics,
        with_tools=with_tools,
        without_tools=without_tools,
        internalization_results=internalization_results,
        consistency_rate=consistency_rate,
        output_dir=config.output_dir,
    )
    logger.info(f"Figures saved → {config.output_dir}/")


def _push_to_hub(model, tokenizer, args: argparse.Namespace) -> None:
    if not args.push_to_hub or not args.hf_repo:
        return
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.warning("HF_TOKEN not set — skipping Hub push")
        return
    logger.info(f"Pushing merged model to HuggingFace Hub: {args.hf_repo}")
    model.push_to_hub_merged(
        args.hf_repo,
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )
    logger.info("Hub push complete ✅")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = build_config(args)

    # ── Phase 0: Setup ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CurricSym-SLM-Lite  ·  RTX 4090 Training Run")
    logger.info("=" * 70)
    set_seed(config.seed)
    log_environment(config, logger)
    logger.info(f"\nVRAM at startup:\n{vram_summary()}")
    _setup_wandb(config)

    # ── Phase 0.1: Data ───────────────────────────────────────────────────
    datasets = build_all_datasets(config)
    train_verified = datasets["train_verified"]
    eval_verified = datasets["eval_verified"]
    sft_train = datasets["sft_train"]
    sft_eval = datasets["sft_eval"]
    grpo_train = datasets["grpo_train"]
    paired_ablation = datasets["paired_ablation"]

    # ── Phase 0.2: Model ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info(f"\nVRAM after model load:\n{vram_summary()}")

    # ── Phase 0.3: Verifier & Curriculum ─────────────────────────────────
    verifier = SymbolicVerifier()
    curriculum = CurriculumScheduler(
        n_stages=config.curriculum_stages,
        target_reward=config.curriculum_target_reward,
        alpha=config.curriculum_alpha,
        eta=config.curriculum_eta,
    )
    curriculum.tool_fade_ratios = list(config.tool_fade_ratios)
    logger.info("Verifier and curriculum scheduler ready ✅")

    # ── Phase 1: SFT Warm-Up ─────────────────────────────────────────────
    run_sft(config, model, tokenizer, sft_train, sft_eval)
    logger.info(f"\nVRAM after SFT:\n{vram_summary()}")

    # ── Phase 2: Staged GRPO RL ───────────────────────────────────────────
    stage_metrics = run_grpo(config, model, tokenizer, grpo_train, curriculum)
    logger.info(f"\nVRAM after GRPO:\n{vram_summary()}")

    # ── Phase 3: Evaluation ───────────────────────────────────────────────
    if args.skip_eval:
        logger.info("--skip_eval set — skipping evaluation phases")
        results = {}
        internalization_results = {}
    else:
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
        internalization_results = results.get("internalization", {})

    # ── Phase 4: Reporting & Visualisation ────────────────────────────────
    if not args.skip_eval and results:
        report = build_experiment_report(
            config=config,
            model=model,
            tokenizer=tokenizer,
            results=results,
            stage_metrics=stage_metrics,
            curriculum=curriculum,
            verifier=verifier,
        )
        _generate_plots(
            config=config,
            stage_metrics=stage_metrics,
            with_tools=results.get("with_tools", {}),
            without_tools=results.get("without_tools", {}),
            internalization_results=internalization_results,
            consistency_rate=results.get("consistency_rate", 0.0),
        )

    # ── Phase 5: Optional model merge & Hub push ──────────────────────────
    if args.merge_model:
        merge_path = str(Path(config.output_dir) / "merged_model")
        merge_and_save_full_model(model, tokenizer, merge_path, quantization="bf16")

    _push_to_hub(model, tokenizer, args)

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("✅  CurricSym-SLM-Lite training pipeline complete")
    logger.info(f"   Outputs     → {config.output_dir}")
    logger.info(f"   Checkpoints → {config.checkpoint_dir}")
    saved = list_checkpoints(config.checkpoint_dir)
    for name, info in saved.items():
        logger.info(f"     {name:<40} {info['size_mb']:>8.1f} MB")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
