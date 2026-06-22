"""
models/model_loader.py — Unsloth model + QLoRA initialisation
==============================================================
Handles model loading, EOS-token patching, optional SFT-adapter
loading, and parameter reporting.  Single-GPU (RTX 4090) path only —
no DDP logic needed here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EOS-token safety patch
# ---------------------------------------------------------------------------
def _ensure_eos_token(tokenizer) -> None:
    """
    Unsloth sometimes sets eos_token to '<EOS_TOKEN>' which is NOT in the
    Qwen2 vocabulary.  This causes TRL to crash during trainer initialisation.
    Patch to the actual chat-template end token.
    """
    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token and tokenizer.eos_token in vocab:
        return  # already valid
    for candidate in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
        if candidate in vocab:
            tokenizer.eos_token = candidate
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(candidate)
            logger.info(f"EOS token patched → '{candidate}' (id={tokenizer.eos_token_id})")
            return
    logger.warning("Could not find a valid EOS token in vocab — proceeding anyway.")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(config, for_inference: bool = False):
    """
    Load base model + tokenizer via Unsloth, apply QLoRA PEFT adapters.

    Parameters
    ----------
    config        : TrainingConfig
    for_inference : bool — if True, skips get_peft_model (eval-only load)

    Returns
    -------
    model, tokenizer
    """
    from unsloth import FastLanguageModel  # noqa: import-errors managed by env

    logger.info(f"Loading: {config.model_name}")
    logger.info(f"  load_in_4bit={config.load_in_4bit}  max_seq_length={config.max_seq_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,            # auto: bfloat16 on Ampere+, float16 otherwise
        load_in_4bit=config.load_in_4bit,
    )

    _ensure_eos_token(tokenizer)
    logger.info(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    if for_inference:
        FastLanguageModel.for_inference(model)
        logger.info("Model set to inference mode (2× faster generation)")
        return model, tokenizer

    # ── Apply QLoRA adapters ──────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        # Unsloth custom GC: best for single-GPU; avoids DDP reentrant issues.
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
    )

    # ── VRAM report ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"VRAM after model load: {alloc:.2f}/{total:.2f} GB")

    _log_trainable_params(model)
    return model, tokenizer


def load_sft_adapter(model, adapter_path: str) -> None:
    """
    Load a previously saved SFT adapter in-place into an existing PeftModel.

    model MUST already be the result of FastLanguageModel.get_peft_model().
    We call model.load_adapter() rather than PeftModel.from_pretrained()
    to avoid the double-wrapping bug.
    """
    if not Path(adapter_path).is_dir():
        logger.warning(f"SFT adapter path not found: {adapter_path} — skipping")
        return
    logger.info(f"Loading SFT adapter from: {adapter_path}")
    model.load_adapter(adapter_path, adapter_name="default")
    logger.info("SFT adapter loaded ✅")


def save_adapter(model, tokenizer, save_path: str, tag: str = "") -> None:
    """Save LoRA adapter weights + tokenizer."""
    p = Path(save_path)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))
    tokenizer.save_pretrained(str(p))
    logger.info(f"{'[' + tag + '] ' if tag else ''}Adapter saved → {p}")


def merge_and_save_full_model(
    model, tokenizer, save_path: str, quantization: str = "bf16"
) -> None:
    """
    Merge LoRA weights into the base model and save as a standalone model.
    Recommended for HuggingFace Hub publishing or vLLM serving.

    quantization: 'bf16' | 'f16' | 'q4_k_m' (requires llama.cpp)
    """
    from unsloth import FastLanguageModel  # noqa

    logger.info(f"Merging LoRA → {quantization} model at: {save_path}")
    if quantization in ("bf16", "f16"):
        model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
    elif quantization == "q4_k_m":
        model.save_pretrained_gguf(save_path, tokenizer, quantization_method="q4_k_m")
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    logger.info("Merge complete ✅")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _log_trainable_params(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({trainable / total * 100:.2f}%)"
    )
