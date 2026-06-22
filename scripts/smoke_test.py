#!/usr/bin/env python3
"""
scripts/smoke_test.py — Fast Smoke Test (no GPU required for imports)
======================================================================
Validates that all modules import correctly and core logic is sane.
Run this before starting a full training job on RunPod.

  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PASS = "✅"
FAIL = "❌"


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        return False


def main() -> None:
    print("=" * 55)
    print("  CurricSym-SLM-Lite — Smoke Test")
    print("=" * 55)

    results = []

    # ── Imports ───────────────────────────────────────────────
    print("\n[Imports]")
    results.append(check("configs.TrainingConfig",
                          lambda: __import__("curricsym.configs", fromlist=["TrainingConfig"])))
    results.append(check("utils.common",
                          lambda: __import__("curricsym.utils", fromlist=["get_logger"])))
    results.append(check("data.loader",
                          lambda: __import__("curricsym.data", fromlist=["build_all_datasets"])))
    results.append(check("models.verifier",
                          lambda: __import__("curricsym.models", fromlist=["SymbolicVerifier"])))
    results.append(check("models.model_loader",
                          lambda: __import__("curricsym.models", fromlist=["load_model_and_tokenizer"])))
    results.append(check("training.reward_functions",
                          lambda: __import__("curricsym.training", fromlist=["grpo_outcome_reward"])))
    results.append(check("training.curriculum",
                          lambda: __import__("curricsym.training", fromlist=["CurriculumScheduler"])))
    results.append(check("training.sft_trainer",
                          lambda: __import__("curricsym.training", fromlist=["run_sft"])))
    results.append(check("training.grpo_trainer",
                          lambda: __import__("curricsym.training", fromlist=["run_grpo"])))
    results.append(check("evaluation.evaluator",
                          lambda: __import__("curricsym.evaluation", fromlist=["run_full_evaluation"])))
    results.append(check("evaluation.visualisation",
                          lambda: __import__("curricsym.evaluation", fromlist=["generate_full_dashboard"])))
    results.append(check("reporting",
                          lambda: __import__("curricsym.reporting", fromlist=["build_experiment_report"])))

    # ── Logic tests ───────────────────────────────────────────
    print("\n[Logic]")

    def _test_verifier():
        from curricsym.models import SymbolicVerifier
        v = SymbolicVerifier()
        assert v.verify("math", "2+2", "4", "4")["correct"]
        assert not v.verify("math", "2+2", "5", "4")["correct"]
        assert v.verify("fol", "q", "True", "True")["correct"]
        assert not v.verify("fol", "q", "True", "False")["correct"]

    def _test_curriculum():
        from curricsym.training import CurriculumScheduler
        sched = CurriculumScheduler(n_stages=3, target_reward=0.5)
        sched.set_difficulty_range(0.0, 5.0)
        for r in [0.3, 0.6, 0.8, 0.9]:
            sched.update(r)
        state = sched.get_state()
        assert "current_stage" in state
        assert 0 <= state["current_stage"] < 3

    def _test_reward_fns():
        from curricsym.training import (
            grpo_outcome_reward, grpo_format_reward,
            grpo_process_reward, grpo_internalization_reward,
        )
        prompts = ["test"]
        completions = ["<thinking>step 1</thinking><answer>4</answer>"]
        r = grpo_format_reward(prompts, completions)
        assert len(r) == 1 and r[0] > 0
        r2 = grpo_internalization_reward(prompts, completions, tool_ratio=[0.0])
        assert r2[0] == 0.1  # no <verify> tag → reward

    def _test_extract():
        from curricsym.training import extract_answer, extract_thinking
        assert extract_answer("<answer>42</answer>") == "42"
        assert extract_thinking("<thinking>hello</thinking>") == "hello"

    def _test_config():
        from curricsym.configs import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.lora_r == 32
        assert cfg.grpo_num_generations == 8
        assert "unsloth" in cfg.model_name

    results.append(check("SymbolicVerifier (math+FOL)",  _test_verifier))
    results.append(check("CurriculumScheduler",           _test_curriculum))
    results.append(check("Reward functions",              _test_reward_fns))
    results.append(check("extract_answer / extract_thinking", _test_extract))
    results.append(check("TrainingConfig defaults",       _test_config))

    # ── Summary ───────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*55}")
    if passed == total:
        print(f"  {PASS}  All {total} checks passed — ready to train!")
    else:
        print(f"  {FAIL}  {passed}/{total} passed — fix errors before training")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
