"""
training/sft_trainer.py — SFT Warm-Up Phase
============================================
Supervised fine-tuning on (prompt, thinking+answer) pairs.
Warm-up phase before GRPO to give the model the correct output
format (thinking/answer tags) and domain vocabulary.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def run_sft(config, model, tokenizer, sft_train, sft_eval) -> None:
    """
    Run the SFT warm-up phase.

    Saves the adapter to config.checkpoint_dir/sft_adapter.
    If config.resume_from_sft is set, skips training entirely.
    """
    from trl import SFTConfig, SFTTrainer

    sft_adapter_path = str(Path(config.checkpoint_dir) / "sft_adapter")

    # ── Skip if resuming ──────────────────────────────────────────────────
    if config.resume_from_sft:
        logger.info(f"SFT skipped — loading existing adapter: {config.resume_from_sft}")
        from ..models.model_loader import load_sft_adapter
        load_sft_adapter(model, config.resume_from_sft)
        return

    logger.info("=" * 60)
    logger.info("PHASE 1: SFT WARM-UP")
    logger.info("=" * 60)
    logger.info(f"  Train examples : {len(sft_train)}")
    logger.info(f"  Eval examples  : {len(sft_eval)}")
    logger.info(f"  Max steps      : {config.sft_max_steps}")
    logger.info(f"  LR             : {config.sft_lr}")
    logger.info(f"  Batch          : {config.sft_batch_size}  GradAccum={config.sft_grad_accum}")

    sft_cfg = SFTConfig(
        output_dir=str(Path(config.output_dir) / "sft"),
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accum,
        learning_rate=config.sft_lr,
        weight_decay=config.sft_weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=config.sft_warmup_ratio,
        max_length=config.max_seq_length,
        logging_steps=10,
        save_steps=config.eval_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_steps=config.sft_max_steps,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name + "_sft" if config.use_wandb else None,
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"append_concat_token": False, "add_special_tokens": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_cfg,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────────
    model.save_pretrained(sft_adapter_path)
    tokenizer.save_pretrained(sft_adapter_path)
    logger.info(f"SFT adapter saved → {sft_adapter_path}")

    # ── VRAM cleanup ───────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    logger.info("SFT complete ✅")
