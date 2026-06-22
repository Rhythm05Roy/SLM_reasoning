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

    # Determine the training dtype so we can re-cast LoRA params after SFT.
    # load_best_model_at_end=True reloads the checkpoint from disk, which
    # restores LoRA B-matrices to float32 (the safetensors default), breaking
    # the dtype alignment needed by Unsloth's fast_lora kernel in GRPO.
    # We disable it and rely on the explicit re-cast below instead.
    # Always use bfloat16 — RTX 5090 Blackwell always supports it.
    _train_dtype = torch.bfloat16

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
        # Do NOT set load_best_model_at_end=True — it reloads the checkpoint
        # from disk and restores LoRA weights to float32, which breaks the
        # dtype alignment required by Unsloth's fast_lora GRPO kernel.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        fp16=False,
        bf16=True,
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

    # ── Re-cast LoRA params to training dtype ─────────────────────────────
    # The SFT Trainer may leave some adapter weights in float32 (e.g. from
    # optimizer state casting or internal HF callbacks). Re-cast here so
    # the model handed to the GRPO stage is fully dtype-consistent.
    recast_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != _train_dtype:
            param.data = param.data.to(_train_dtype)
            recast_count += 1
    if recast_count:
        logger.info(
            f"Re-cast {recast_count} LoRA param tensors to {_train_dtype} "
            "after SFT (prevents GRPO fast_lora dtype clash)"
        )

    # ── Save adapter ───────────────────────────────────────────────────────
    model.save_pretrained(sft_adapter_path)
    tokenizer.save_pretrained(sft_adapter_path)
    logger.info(f"SFT adapter saved → {sft_adapter_path}")

    # ── VRAM cleanup ───────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    logger.info("SFT complete ✅")
