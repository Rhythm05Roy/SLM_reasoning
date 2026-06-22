"""
training/curriculum.py — AdaRFT-Inspired Curriculum Scheduler
==============================================================
Dynamically adjusts training difficulty and tool availability based
on the model's recent reward history.

Algorithm (AdaRFT-style):
    difficulty_target += alpha * (avg_reward - target_reward) * eta

  avg_reward > target_reward  → increase difficulty
  avg_reward < target_reward  → decrease difficulty

Tool-fading follows from the current stage (percentile of difficulty_target
within [min, max]), transitioning:
  Stage 0: tool_ratio = 1.0  (full symbolic access)
  Stage 1: tool_ratio = 0.5  (partial access + internalization reward)
  Stage 2: tool_ratio = 0.0  (no tools — full internalization test)
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Adaptive difficulty scheduler.

    Parameters
    ----------
    n_stages       : number of curriculum stages
    target_reward  : target mean reward (default 0.5)
    alpha          : difficulty update rate
    eta            : difficulty update scale
    """

    def __init__(
        self,
        n_stages: int = 3,
        target_reward: float = 0.5,
        alpha: float = 2.0,
        eta: float = 50.0,
    ):
        self.n_stages = n_stages
        self.target_reward = target_reward
        self.alpha = alpha
        self.eta = eta
        self.current_stage: int = 0
        self.reward_history: List[float] = []
        self.difficulty_target: float = 0.0
        self.step_count: int = 0
        self.min_difficulty: float = 0.0
        self.max_difficulty: float = 5.0
        self.tool_fade_ratios: List[float] = [1.0, 0.5, 0.0]

    def set_difficulty_range(self, lo: float, hi: float) -> None:
        self.min_difficulty = lo
        self.max_difficulty = hi
        self.difficulty_target = (lo + hi) / 2.0

    def update(self, reward: float) -> None:
        """Update difficulty target based on most recent reward."""
        self.reward_history.append(reward)
        self.step_count += 1
        window = min(20, len(self.reward_history))
        avg = float(np.mean(self.reward_history[-window:]))
        self.difficulty_target += self.alpha * (avg - self.target_reward) * self.eta
        self.difficulty_target = float(
            np.clip(self.difficulty_target, self.min_difficulty, self.max_difficulty)
        )
        progress = (self.difficulty_target - self.min_difficulty) / max(
            self.max_difficulty - self.min_difficulty, 1e-6
        )
        self.current_stage = min(int(progress * self.n_stages), self.n_stages - 1)

    def get_tool_ratio(self) -> float:
        idx = min(self.current_stage, len(self.tool_fade_ratios) - 1)
        return self.tool_fade_ratios[idx]

    def sample_indices(
        self,
        difficulties: list,
        n_samples: int,
        strategy: str = "mixed",
    ) -> list:
        """Return indices sampled according to current difficulty target."""
        diffs = np.array(difficulties)
        n_total = len(diffs)

        if strategy == "closest":
            dist = np.abs(diffs - self.difficulty_target)
            weights = np.exp(-dist / (np.std(diffs) + 1e-6))
            weights /= weights.sum()
            return np.random.choice(
                n_total, size=min(n_samples, n_total), replace=False, p=weights
            ).tolist()

        elif strategy == "mixed":
            n_close = int(0.7 * n_samples)
            n_rand = n_samples - n_close
            dist = np.abs(diffs - self.difficulty_target)
            weights = np.exp(-dist / (np.std(diffs) + 1e-6))
            weights /= weights.sum()
            close = np.random.choice(n_total, size=min(n_close, n_total),
                                     replace=False, p=weights)
            rand_idx = np.random.choice(n_total, size=min(n_rand, n_total), replace=False)
            return np.concatenate([close, rand_idx]).tolist()

        else:  # random fallback
            return np.random.choice(
                n_total, size=min(n_samples, n_total), replace=False
            ).tolist()

    def get_state(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "difficulty_target": self.difficulty_target,
            "avg_reward": (
                float(np.mean(self.reward_history[-20:])) if self.reward_history else 0.0
            ),
            "tool_ratio": self.get_tool_ratio(),
            "step_count": self.step_count,
        }

    def reset(self) -> None:
        """Reset state (preserves hyper-parameters)."""
        self.current_stage = 0
        self.reward_history = []
        self.difficulty_target = (self.min_difficulty + self.max_difficulty) / 2.0
        self.step_count = 0


# ---------------------------------------------------------------------------
# TrainerCallback that wires the scheduler into GRPOTrainer
# ---------------------------------------------------------------------------
class CurriculumCallback(TrainerCallback):
    """
    Reads logged rewards from GRPOTrainer → updates CurriculumScheduler.

    GRPOTrainer logs 'reward/mean' (TRL ≥ 0.12) or 'rewards/mean' (older).
    We try both keys and fall back to 0.5 if neither is present.
    """

    def __init__(self, scheduler: CurriculumScheduler):
        self.scheduler = scheduler

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        reward = (
            logs.get("reward/mean")
            or logs.get("rewards/mean")
            or logs.get("reward")
            or 0.5
        )
        self.scheduler.update(float(reward))
        if state.global_step % 25 == 0:
            s = self.scheduler.get_state()
            logger.info(
                f"[Curriculum] step={state.global_step:4d}  "
                f"stage={s['current_stage']}  "
                f"diff={s['difficulty_target']:.2f}  "
                f"tool_ratio={s['tool_ratio']:.1f}  "
                f"avg_r={s['avg_reward']:.3f}"
            )
