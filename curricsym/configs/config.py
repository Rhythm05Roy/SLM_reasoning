"""
CurricSym-SLM-Lite  ·  Master Configuration
============================================================
Tuned for RunPod RTX 4090 (24 GB VRAM, single-GPU).

Key differences vs. Kaggle T4 baseline
  • load_in_4bit = False  → bfloat16 full-rank (4090 has headroom)
  • Larger batches / more generations
  • Longer max_seq_length
  • use_gradient_checkpointing = "unsloth"  (single-GPU, no DDP conflicts)
  • W&B enabled by default
"""

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Helper — resolve env-var overrides (useful for RunPod secrets)
# ---------------------------------------------------------------------------
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class TrainingConfig:
    """Master configuration for CurricSym-SLM-Lite (RTX 4090 profile)."""

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    # Alternatives:
    #   "unsloth/Llama-3.2-3B-Instruct"
    #   "unsloth/Phi-3.5-mini-instruct"
    #   "unsloth/Qwen2.5-7B-Instruct"  (fits in 24 GB BF16)
    max_seq_length: int = 4096      # 4090: 24 GB → comfortable at 4096
    load_in_4bit: bool = False      # BF16 on 4090; set True for 7B+ models

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 32                # larger rank = more expressive (4090 allows)
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"  # single-GPU: Unsloth custom GC

    # ── SFT Warm-Up ────────────────────────────────────────────────────────
    sft_lr: float = 2e-4
    sft_batch_size: int = 4         # per-device; effective = 4 * 8 = 32
    sft_grad_accum: int = 8
    sft_max_steps: int = 500
    sft_weight_decay: float = 0.01
    sft_warmup_ratio: float = 0.05

    # ── GRPO RL ────────────────────────────────────────────────────────────
    grpo_lr: float = 5e-6
    grpo_batch_size: int = 8        # 4090: generous; must equal num_generations
    grpo_grad_accum: int = 1        # must be 1 for GRPO
    grpo_num_generations: int = 8   # 4090: double T4 value
    grpo_max_steps: int = 300       # total steps; split evenly across 3 stages
    grpo_temperature: float = 1.0
    grpo_beta: float = 0.04         # KL penalty
    grpo_epsilon: float = 0.2       # clip ratio
    grpo_weight_decay: float = 0.01

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum_stages: int = 3
    curriculum_target_reward: float = 0.5
    curriculum_alpha: float = 2.0   # difficulty update rate
    curriculum_eta: float = 50.0    # difficulty update scale
    tool_fade_ratios: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.0])

    # ── Internalization ────────────────────────────────────────────────────
    internalization_weight: float = 0.3
    consistency_temperature: float = 2.0

    # ── Data ───────────────────────────────────────────────────────────────
    gsm_symbolic_size: int = 3000
    proofwriter_size: int = 2000
    eval_split_ratio: float = 0.1   # fraction of combined data used for eval

    # ── Paths (RunPod) ─────────────────────────────────────────────────────
    # These can be overridden via environment variables
    output_dir: str = field(
        default_factory=lambda: _env("OUTPUT_DIR", "/workspace/curricsym_output"))
    checkpoint_dir: str = field(
        default_factory=lambda: _env("CHECKPOINT_DIR", "/workspace/curricsym_checkpoints"))
    data_cache_dir: str = field(
        default_factory=lambda: _env("DATA_CACHE_DIR", "/workspace/curricsym_data"))

    # ── Weights & Biases ───────────────────────────────────────────────────
    wandb_project: str = "curricsym-slm-lite"
    wandb_run_name: str = ""        # auto-generated if empty
    use_wandb: bool = True          # enable on RunPod

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_steps: int = 50
    max_eval_examples: int = 300    # more than Kaggle; 4090 is fast at inference
    internalization_eval_examples: int = 300

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_from_sft: str = ""       # path to SFT adapter to skip SFT phase
    resume_from_grpo_stage: int = 0 # which GRPO stage to resume from (0 = start)

    def __post_init__(self):
        import os
        for d in [self.output_dir, self.checkpoint_dir, self.data_cache_dir]:
            os.makedirs(d, exist_ok=True)
        if not self.wandb_run_name:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            model_tag = self.model_name.split("/")[-1].replace("-Instruct", "")
            self.wandb_run_name = f"curricsym_{model_tag}_{ts}"
