"""training/curriculum.py — AdaRFT-style curriculum scheduler."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Adaptive difficulty scheduler (AdaRFT-inspired).

    Algorithm:
        difficulty_target += alpha * (avg_reward - target_reward) * eta

    avg_reward > target → increase difficulty
    avg_reward < target → decrease difficulty
    Stage transitions are proportional to progress in [min, max] difficulty.
    """

    def __init__(self, n_stages: int = 3, target_reward: float = 0.5,
                 alpha: float = 2.0, eta: float = 50.0):
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

    def get_state(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "difficulty_target": self.difficulty_target,
            "avg_reward": float(np.mean(self.reward_history[-20:])) if self.reward_history else 0.0,
            "tool_ratio": self.get_tool_ratio(),
            "step_count": self.step_count,
        }

    def reset(self) -> None:
        self.current_stage = 0
        self.reward_history = []
        self.difficulty_target = (self.min_difficulty + self.max_difficulty) / 2.0
        self.step_count = 0


class CurriculumCallback(TrainerCallback):
    """Wires CurriculumScheduler into GRPOTrainer via on_log hook."""

    def __init__(self, scheduler: CurriculumScheduler):
        self.scheduler = scheduler

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs: dict | None = None, **kwargs) -> None:
        if logs is None:
            return
        reward = (logs.get("reward/mean") or logs.get("rewards/mean")
                  or logs.get("reward") or 0.5)
        self.scheduler.update(float(reward))
        if state.global_step % 25 == 0:
            s = self.scheduler.get_state()
            logger.info(
                f"[Curriculum] step={state.global_step:4d}  stage={s['current_stage']}  "
                f"diff={s['difficulty_target']:.2f}  tool_ratio={s['tool_ratio']:.1f}  "
                f"avg_r={s['avg_reward']:.3f}"
            )
