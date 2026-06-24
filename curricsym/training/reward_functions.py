"""training/reward_functions.py — 4 GRPO reward signals."""
from __future__ import annotations

import re
from typing import List

from ..models.verifier import SymbolicVerifier

_verifier = SymbolicVerifier()


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


def grpo_outcome_reward(
    prompts: List[str], completions: List[str],
    *, answer: List[str] | None = None,
    domain: List[str] | None = None, **kwargs,
) -> List[float]:
    """Z3 / string correctness: +1.0 correct, -1.0 wrong, -0.5 no answer."""
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
    prompts: List[str], completions: List[str], **kwargs,
) -> List[float]:
    """<thinking>/<answer> structural compliance. Max = 1.0."""
    rewards = []
    for c in completions:
        s = 0.0
        if "<thinking>" in c and "</thinking>" in c:
            s += 0.3
        if "<answer>" in c and "</answer>" in c:
            s += 0.3
        tp, ap = c.find("<thinking>"), c.find("<answer>")
        if tp >= 0 and ap > tp:
            s += 0.2
        if len(extract_thinking(c)) > 20:
            s += 0.2
        rewards.append(s)
    return rewards


def grpo_process_reward(
    prompts: List[str], completions: List[str],
    *, domain: List[str] | None = None, **kwargs,
) -> List[float]:
    """
    Heuristic PRM: step-by-step quality signal.
    Credits thinking in sweet-spot length, step markers, verify token.

    Thesis note: This is a heuristic baseline PRM.
    Full neural PRM distillation from verifier traces is future work.
    """
    if domain is None:
        domain = ["math"] * len(prompts)
    rewards = []
    for c, d in zip(completions, domain):
        thinking = extract_thinking(c)
        s = 0.0
        tl = len(thinking)
        if 50 < tl < 600:
            s += 0.3
        if any(m in thinking.lower() for m in ["step", "first", "then", "next", "therefore"]):
            s += 0.1
        if d == "math" and "<verify>" in c:
            s += 0.2
        rewards.append(min(s, 1.0))
    return rewards


def grpo_internalization_reward(
    prompts: List[str], completions: List[str],
    *, tool_ratio: List[float] | None = None, **kwargs,
) -> List[float]:
    """
    Internalization signal: penalise tool use when tool_ratio < 0.3.
    Encourages model to internalise symbolic reasoning in late stages.

    Returns:
      -0.3  if used <verify> when tool_ratio < 0.3  (penalise dependency)
      +0.1  if NOT used <verify> when tool_ratio < 0.3  (reward independence)
       0.0  otherwise (tool enabled, no signal)
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


def get_reward_functions(include_internalization: bool = False) -> list:
    fns = [grpo_outcome_reward, grpo_format_reward, grpo_process_reward]
    if include_internalization:
        fns.append(grpo_internalization_reward)
    return fns


def get_verifier() -> SymbolicVerifier:
    return _verifier
