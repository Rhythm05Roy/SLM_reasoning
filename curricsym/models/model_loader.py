"""
models/model_loader.py — Unsloth model + QLoRA initialisation (single-GPU).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _ensure_eos_token(tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token and tokenizer.eos_token in vocab:
        return
    for candidate in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
        if candidate in vocab:
            tokenizer.eos_token = candidate
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(candidate)
            logger.info(f"EOS token patched → '{candidate}' (id={tokenizer.eos_token_id})")
            return
    logger.warning("Could not find a valid EOS token in vocab — proceeding anyway.")


def load_model_and_tokenizer(config, for_inference: bool = False):
    from unsloth import FastLanguageModel

    logger.info(f"Loading: {config.model_name}")
    logger.info(f"  load_in_4bit={config.load_in_4bit}  max_seq_length={config.max_seq_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    _ensure_eos_token(tokenizer)
    logger.info(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    if for_inference:
        FastLanguageModel.for_inference(model)
        logger.info("Model set to inference mode")
        return model, tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
    )

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"VRAM after model load: {alloc:.2f}/{total:.2f} GB")

    _log_trainable_params(model)
    return model, tokenizer


def load_sft_adapter(model, adapter_path: str) -> None:
    if not Path(adapter_path).is_dir():
        logger.warning(f"SFT adapter path not found: {adapter_path} — skipping")
        return
    logger.info(f"Loading SFT adapter from: {adapter_path}")
    model.load_adapter(adapter_path, adapter_name="default")
    logger.info("SFT adapter loaded ✅")


def save_adapter(model, tokenizer, save_path: str, tag: str = "") -> None:
    p = Path(save_path)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))
    tokenizer.save_pretrained(str(p))
    logger.info(f"{'[' + tag + '] ' if tag else ''}Adapter saved → {p}")


def merge_and_save_full_model(model, tokenizer, save_path: str, quantization: str = "bf16") -> None:
    logger.info(f"Merging LoRA → {quantization} at: {save_path}")
    if quantization in ("bf16", "f16"):
        model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
    elif quantization == "q4_k_m":
        model.save_pretrained_gguf(save_path, tokenizer, quantization_method="q4_k_m")
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    logger.info("Merge complete ✅")


def _log_trainable_params(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
