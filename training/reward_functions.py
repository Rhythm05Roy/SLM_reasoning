"""
training/reward_functions.py — GRPO Reward Functions
=====================================================
Four reward signals used during GRPO training.

TRL's GRPOTrainer passes dataset columns as keyword arguments using their
*exact column names*.  All signatures use keyword-only parameters to match
the column names in grpo_train:  answer, domain, tool_ratio.

Reward weights (from thesis):
  outcome      1.0  (primary signal — correctness)
  format       0.3  (structural compliance)
  process      0.3  (heuristic PRM — step-by-step quality)
  internalize  0.2  (tool-fading signal — penalise tool use in late stages)
"""
from __future__ import annotations

import re
from typing import List

from ..models.verifier import SymbolicVerifier

# Module-level verifier — shared across all reward functions in a process
_verifier = SymbolicVerifier()


# ---------------------------------------------------------------------------
# Answer / thinking extraction helpers
# ---------------------------------------------------------------------------
def extract_answer(completion: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(-?[\d,\.]+)", completion)
    if m:
        return m.group(1).strip()
    lines = completion.strip().split("\n")
    return lines[-1].strip() if lines else ""


def extract_thinking(completion: str) -> str:
    m = re.search(r"<thinking>(.*?)</thinking>", completion, re.DOTALL)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def grpo_outcome_reward(
    prompts: List[str],
    completions: List[str],
    *,
    answer: List[str] | None = None,
    domain: List[str] | None = None,
    **kwargs,
) -> List[float]:
    """
    Outcome reward: Z3 / string correctness.
    Returns +1.0 if correct, -1.0 if wrong, -0.5 if no answer found.
    """
    if answer is None:
        answer = [""] * len(prompts)
    if domain is None:
        domain = ["math"] * len(prompts)
    rewards = []
    for p, c, a, d in zip(prompts, completions, answer, domain):
        pred = extract_answer(c)
        if not pred:
            rewards.append(-0.5)
            continue
        result = _verifier.verify(d, str(p), pred, str(a))
        rewards.append(1.0 if result["correct"] else -1.0)
    return rewards


def grpo_format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Format reward: <thinking>...</thinking><answer>...</answer> structure.
    Max score = 1.0.
    """
    rewards = []
    for c in completions:
        s = 0.0
        if "<thinking>" in c and "</thinking>" in c:
            s += 0.3
        if "<answer>" in c and "</answer>" in c:
            s += 0.3
        tp = c.find("<thinking>")
        ap = c.find("<answer>")
        if tp >= 0 and ap > tp:
            s += 0.2
        if len(extract_thinking(c)) > 20:
            s += 0.2
        rewards.append(s)
    return rewards


def grpo_process_reward(
    prompts: List[str],
    completions: List[str],
    *,
    domain: List[str] | None = None,
    **kwargs,
) -> List[float]:
    """
    Heuristic Process Reward Model (PRM): step-by-step quality signal.
    Credits thinking length in sweet-spot range, step markers, and
    explicit verify tokens when tools are enabled.

    Thesis note: this is a *heuristic* PRM baseline.  Full neural PRM
    distillation (from verifier-certified traces) is left as future work.
    """
    if domain is None:
        domain = ["math"] * len(prompts)
    rewards = []
    for c, d in zip(completions, domain):
        thinking = extract_thinking(c)
        s = 0.0
        tl = len(thinking)
        if 50 < tl < 800:
            s += 0.3
        step_markers = ["step", "first", "then", "next", "therefore"]
        if any(m in thinking.lower() for m in step_markers):
            s += 0.1
        if d == "math" and "<verify>" in c:
            s += 0.2  # credit for using tool when allowed
        rewards.append(min(s, 1.0))
    return rewards


def grpo_internalization_reward(
    prompts: List[str],
    completions: List[str],
    *,
    tool_ratio: List[float] | None = None,
    **kwargs,
) -> List[float]:
    """
    Internalization reward: penalise tool use in faded (low tool_ratio) stages.
    Encourages the model to internalise symbolic reasoning rather than
    relying on external verifier calls.

    Returns:
       -0.3 if model used <verify> when tool_ratio < 0.3  (penalise)
       +0.1 if model did NOT use <verify> when tool_ratio < 0.3  (reward)
        0.0 otherwise (tool still enabled, no signal)
    """
    if tool_ratio is None:
        tool_ratio = [0.5] * len(prompts)
    rewards = []
    for c, tr in zip(completions, tool_ratio):
        has_verify = "<verify>" in c
        if tr < 0.3:
            rewards.append(-0.3 if has_verify else 0.1)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Composite reward list for each GRPO stage
# ---------------------------------------------------------------------------
def get_reward_functions(include_internalization: bool = False) -> list:
    fns = [grpo_outcome_reward, grpo_format_reward, grpo_process_reward]
    if include_internalization:
        fns.append(grpo_internalization_reward)
    return fns


def get_verifier() -> SymbolicVerifier:
    """Return the module-level verifier (shared singleton)."""
    return _verifier
