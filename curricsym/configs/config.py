"""
CurricSym-SLM-Lite  ·  Master Configuration
============================================================
Budget-optimised profile for RunPod RTX 4090 (24 GB VRAM).

Target: complete SFT + 3-stage GRPO + full eval in ≤ 2 hours,
costing ≤ $1.50 on Community Cloud ($0.34/hr).

Model: Qwen2.5-1.5B-Instruct (default)
  → 1.5B uses ~4 GB base VRAM, ~11 GB total with GRPO 4 rollouts
  → Leaves 13 GB headroom; safe even with flash-attn disabled

Fallback (if reviewers want larger):
  model_name = "unsloth/Qwen2.5-3B-Instruct"
  load_in_4bit = True
  grpo_num_generations = 4   (keep at 4 for 3B)
  → ~3 hours, ~$1.70
"""

import os
from dataclasses import dataclass, field
from typing import List


def _env(name: str, default: str) -> str:
    """Read optional RunPod env-var override."""
    return os.environ.get(name, default)


@dataclass
class TrainingConfig:
    """Master configuration — budget-safe RTX 4090 profile."""

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"
    # Fallback: "unsloth/Qwen2.5-3B-Instruct" (set load_in_4bit=True)
    max_seq_length: int = 2048      # 2048 saves ~40% VRAM vs 4096
    load_in_4bit: bool = False      # BF16 is fine for 1.5B on 24 GB

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 16                # 16 → fast convergence, less VRAM than 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"

    # ── SFT Warm-Up ────────────────────────────────────────────────────────
    sft_lr: float = 2e-4
    sft_batch_size: int = 4
    sft_grad_accum: int = 8         # effective batch = 32
    sft_max_steps: int = 300        # ~20 min on 1.5B
    sft_weight_decay: float = 0.01
    sft_warmup_ratio: float = 0.05

    # ── GRPO RL ────────────────────────────────────────────────────────────
    grpo_lr: float = 5e-6
    grpo_batch_size: int = 4        # MUST equal num_generations
    grpo_grad_accum: int = 1        # must be 1 for GRPO
    grpo_num_generations: int = 4   # 4 rollouts → safe on 24 GB with 2048 seq
    grpo_max_steps: int = 200       # ~66 steps per stage, ~25 min each
    grpo_temperature: float = 1.0
    grpo_beta: float = 0.04         # KL penalty
    grpo_epsilon: float = 0.2       # clip ratio
    grpo_weight_decay: float = 0.01

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum_stages: int = 3
    curriculum_target_reward: float = 0.5
    curriculum_alpha: float = 2.0
    curriculum_eta: float = 50.0
    tool_fade_ratios: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.0])

    # ── Internalization ────────────────────────────────────────────────────
    internalization_weight: float = 0.3
    consistency_temperature: float = 2.0

    # ── Data ───────────────────────────────────────────────────────────────
    gsm_symbolic_size: int = 2000   # reduced from 3000 for speed
    proofwriter_size: int = 1500    # reduced from 2000 for speed
    eval_split_ratio: float = 0.1

    # ── Paths (RunPod) ─────────────────────────────────────────────────────
    output_dir: str = field(
        default_factory=lambda: _env("OUTPUT_DIR", "/workspace/curricsym_output"))
    checkpoint_dir: str = field(
        default_factory=lambda: _env("CHECKPOINT_DIR", "/workspace/curricsym_checkpoints"))
    data_cache_dir: str = field(
        default_factory=lambda: _env("DATA_CACHE_DIR", "/workspace/curricsym_data"))

    # ── Weights & Biases ───────────────────────────────────────────────────
    wandb_project: str = "curricsym-slm-lite"
    wandb_run_name: str = ""
    use_wandb: bool = True

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_steps: int = 50
    max_eval_examples: int = 150    # reduced from 300 → saves ~10 min
    internalization_eval_examples: int = 150

    # ── OOD Evaluation ─────────────────────────────────────────────────────
    run_ood_eval: bool = True       # Evaluate on GSM-Symbolic p1 (harder)
    ood_eval_examples: int = 100

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_from_sft: str = ""
    resume_from_grpo_stage: int = 0

    def __post_init__(self):
        import os
        from pathlib import Path
        
        # Test paths and fallback to project root subdirs if write permission is denied
        test_dirs = [self.output_dir, self.checkpoint_dir, self.data_cache_dir]
        try:
            for d in test_dirs:
                os.makedirs(d, exist_ok=True)
        except (PermissionError, OSError):
            # Fallback to project root directory outputs/checkpoints/data
            project_root = Path(__file__).resolve().parent.parent.parent
            self.output_dir = str(project_root / "outputs")
            self.checkpoint_dir = str(project_root / "checkpoints")
            self.data_cache_dir = str(project_root / "data_cache")
            for d in [self.output_dir, self.checkpoint_dir, self.data_cache_dir]:
                os.makedirs(d, exist_ok=True)

        if not self.wandb_run_name:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            model_tag = self.model_name.split("/")[-1].replace("-Instruct", "")
            self.wandb_run_name = f"curricsym_{model_tag}_{ts}"
