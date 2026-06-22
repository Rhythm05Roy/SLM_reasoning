"""
training/grpo_trainer.py — Staged GRPO Reinforcement Learning
=============================================================
Three-stage curriculum GRPO training on RTX 4090.

Stage 0  tool_ratio=1.0   reward_fns=[outcome, format, process]
Stage 1  tool_ratio=0.5   reward_fns=[outcome, format, process, internalize]
Stage 2  tool_ratio=0.0   reward_fns=[outcome, format, process, internalize]

Each stage:
  • Overrides tool_ratio column in a copy of the dataset
  • Trains for grpo_max_steps // 3 steps
  • Saves adapter to checkpoint_dir/grpo_stage_{i}
  • Clears VRAM between stages

The CurriculumCallback wires the CurriculumScheduler into each trainer,
enabling AdaRFT-style dynamic difficulty adjustment per step.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper — override tool_ratio for a stage
# ---------------------------------------------------------------------------
def _stage_dataset(base: Dataset, tool_ratio: float) -> Dataset:
    def _override(ex):
        return {**ex, "tool_ratio": tool_ratio}
    return base.map(_override)


# ---------------------------------------------------------------------------
# Single stage trainer
# ---------------------------------------------------------------------------
def _run_stage(
    stage: int,
    tool_ratio: float,
    steps: int,
    config,
    model,
    tokenizer,
    grpo_dataset: Dataset,
    curriculum_callback,
    reward_fns: List,
) -> dict:
    from trl import GRPOConfig, GRPOTrainer

    logger.info("\n" + "=" * 50)
    logger.info(f"GRPO Stage {stage}  tool_ratio={tool_ratio}  steps={steps}")
    logger.info("=" * 50)

    stage_data = _stage_dataset(
        grpo_dataset.select(range(min(len(grpo_dataset), 1500))),
        tool_ratio,
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(Path(config.output_dir) / f"grpo_stage_{stage}"),
        learning_rate=config.grpo_lr * (0.5 ** stage),  # decay per stage
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=config.grpo_weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=config.grpo_batch_size,
        gradient_accumulation_steps=config.grpo_grad_accum,
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        num_generations=config.grpo_num_generations,
        max_prompt_length=config.max_seq_length // 2,
        max_completion_length=config.max_seq_length // 2,
        temperature=config.grpo_temperature,
        beta=config.grpo_beta,
        epsilon=config.grpo_epsilon,
        num_iterations=1,
        max_steps=steps,
        logging_steps=5,
        save_steps=steps,
        save_total_limit=1,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=(
            f"{config.wandb_run_name}_grpo_stage{stage}" if config.use_wandb else None
        ),
        remove_unused_columns=False,
        # Single-GPU RTX 4090: no DDP, so these are not needed, but safe to set
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=grpo_cfg,
        train_dataset=stage_data,
        callbacks=[curriculum_callback],
    )

    result = trainer.train()

    # Save stage adapter
    stage_ckpt = str(Path(config.checkpoint_dir) / f"grpo_stage_{stage}")
    model.save_pretrained(stage_ckpt)
    tokenizer.save_pretrained(stage_ckpt + "_tokenizer")
    logger.info(f"Stage {stage} adapter saved → {stage_ckpt}")
    logger.info(f"Stage {stage} done — loss={result.training_loss:.4f}")

    torch.cuda.empty_cache()
    return {
        "stage": stage,
        "tool_ratio": tool_ratio,
        "loss": result.training_loss,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Main entry — three-stage pipeline
# ---------------------------------------------------------------------------
def run_grpo(
    config,
    model,
    tokenizer,
    grpo_dataset: Dataset,
    curriculum,
) -> List[dict]:
    """
    Run all three GRPO stages.  Returns list of per-stage metric dicts.

    Parameters
    ----------
    config       : TrainingConfig
    model        : PeftModel from load_model_and_tokenizer
    tokenizer    : AutoTokenizer
    grpo_dataset : Dataset with columns [prompt, answer, domain, difficulty, tool_ratio]
    curriculum   : CurriculumScheduler (updated per step via callback)
    """
    from .curriculum import CurriculumCallback
    from .reward_functions import get_reward_functions

    logger.info("=" * 60)
    logger.info("PHASE 2: STAGED GRPO TRAINING")
    logger.info("=" * 60)

    # Set difficulty range from dataset
    all_diffs = grpo_dataset["difficulty"]
    curriculum.set_difficulty_range(min(all_diffs), max(all_diffs))

    steps_per_stage = config.grpo_max_steps // 3
    stage_metrics: List[dict] = []

    stage_configs = [
        (0, 1.0, False),    # stage, tool_ratio, include_internalization
        (1, 0.5, True),
        (2, 0.0, True),
    ]

    # Support resuming from a specific stage
    start_stage = config.resume_from_grpo_stage

    for stage, tool_ratio, include_intern in stage_configs:
        if stage < start_stage:
            logger.info(f"Skipping stage {stage} (resuming from stage {start_stage})")
            continue

        curriculum_callback = CurriculumCallback(curriculum)
        reward_fns = get_reward_functions(include_internalization=include_intern)

        metrics = _run_stage(
            stage=stage,
            tool_ratio=tool_ratio,
            steps=steps_per_stage,
            config=config,
            model=model,
            tokenizer=tokenizer,
            grpo_dataset=grpo_dataset,
            curriculum_callback=curriculum_callback,
            reward_fns=reward_fns,
        )
        stage_metrics.append(metrics)

    # ── Save final adapter ─────────────────────────────────────────────────
    final_path = str(Path(config.checkpoint_dir) / "grpo_final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path + "_tokenizer")
    logger.info(f"Final GRPO adapter saved → {final_path}")

    # ── Persist stage metrics ──────────────────────────────────────────────
    metrics_path = str(Path(config.output_dir) / "grpo_stage_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(stage_metrics, f, indent=2)
    logger.info(f"Stage metrics saved → {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("GRPO COMPLETE")
    for m in stage_metrics:
        logger.info(
            f"  Stage {m['stage']}  tool_ratio={m['tool_ratio']}  loss={m['loss']:.4f}"
        )
    logger.info(f"Curriculum final: {curriculum.get_state()}")
    logger.info("=" * 60)

    return stage_metrics
