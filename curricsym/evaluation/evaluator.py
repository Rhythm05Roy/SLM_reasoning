"""
evaluation/evaluator.py — Evaluation Framework & Ablation Studies
=================================================================
Implements all metrics reported in the thesis:

  EvaluationFramework
    • overall_accuracy, math_accuracy, fol_accuracy
    • avg_faithfulness  (heuristic PRM alignment)
    • avg_format_score  (<thinking>/<answer> compliance)
    • avg_latency_s     (per-example inference time)
    • tool_call_rate    (fraction of completions using <verify>)

  InternalizationEvaluator
    • Paired with-tool / without-tool accuracy
    • internalization_delta  (lower = better internalization)
    • consistency_rate       (same answer with/without tools)

  run_ablation_studies
    • Ablation 1: Curriculum stage progression (AdaRFT vs static)
    • Ablation 2: Tool reliance analysis
    • Ablation 3: Domain-specific performance (math vs FOL)
    • Ablation 4: Process faithfulness
    • Ablation 5: Efficiency metrics
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared generation helper
# ---------------------------------------------------------------------------
def _generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Single-example inference with VRAM-safe cleanup."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    del inputs, out
    return completion


# ---------------------------------------------------------------------------
# EvaluationFramework
# ---------------------------------------------------------------------------
class EvaluationFramework:
    """
    Full evaluation suite: accuracy, faithfulness, format, latency.

    Used for both with-tools and without-tools evaluation to compute
    the internalization delta reported in the thesis results table.
    """

    def __init__(self, model, tokenizer, verifier, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def _faithfulness(self, thinking: str, vresult: dict) -> float:
        """Heuristic: does the thinking trace align with the verifier result?"""
        if not thinking:
            return 0.0
        s = 0.0
        if any(w in thinking.lower() for w in ["verify", "check"]):
            s += 0.3
        if vresult["correct"] and any(
            w in thinking.lower() for w in ["therefore", "so", "thus", "answer"]
        ):
            s += 0.4
        if 50 < len(thinking) < 500:
            s += 0.3
        return min(s, 1.0)

    def _format_score(self, completion: str) -> float:
        """Score structural compliance with <thinking>/<answer> format."""
        s = 0.0
        if "<thinking>" in completion and "</thinking>" in completion:
            s += 0.3
        if "<answer>" in completion and "</answer>" in completion:
            s += 0.3
        tp = completion.find("<thinking>")
        ap = completion.find("<answer>")
        if 0 <= tp < ap:
            s += 0.2
        from ..training.reward_functions import extract_thinking
        if len(extract_thinking(completion)) > 20:
            s += 0.2
        return s

    def evaluate(
        self,
        dataset: Dataset,
        max_examples: int = 200,
        use_tools: bool = True,
        tag: str = "",
    ) -> dict:
        """
        Run evaluation loop.

        Returns
        -------
        dict with keys: overall_accuracy, math_accuracy, fol_accuracy,
                        avg_faithfulness, avg_format_score, avg_latency_s,
                        tool_call_rate, n_examples
        """
        from ..data.loader import format_prompt_with_tools
        from ..training.reward_functions import extract_answer, extract_thinking

        n = min(max_examples, len(dataset))
        subset = dataset.select(range(n))
        label = f"[{'WITH' if use_tools else 'NO'} TOOLS{' ' + tag if tag else ''}]"

        acc_all, math_acc, fol_acc = [], [], []
        faith, fmt, lat = [], [], []
        tool_calls = 0

        self.model.eval()
        logger.info(f"  {label} evaluating {n} examples...")

        for i, ex in enumerate(subset):
            prompt = format_prompt_with_tools(
                ex,
                include_tools=use_tools,
                tool_ratio=1.0 if use_tools else 0.0,
            )
            t0 = time.time()
            completion = _generate_one(
                self.model, self.tokenizer, prompt, self.max_new_tokens
            )
            lat.append(time.time() - t0)

            pred = extract_answer(completion)
            vresult = self.verifier.verify(
                ex["domain"], ex["prompt"], pred, ex["answer"]
            )
            ok = float(vresult["correct"])
            acc_all.append(ok)
            (math_acc if ex["domain"] == "math" else fol_acc).append(ok)

            thinking = extract_thinking(completion)
            faith.append(self._faithfulness(thinking, vresult))
            fmt.append(self._format_score(completion))
            if "<verify>" in completion:
                tool_calls += 1

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                logger.info(
                    f"    {label} [{i+1}/{n}]  acc={np.mean(acc_all):.3f}"
                )

        self.model.train()
        return {
            "overall_accuracy": float(np.mean(acc_all)) if acc_all else 0.0,
            "math_accuracy": float(np.mean(math_acc)) if math_acc else 0.0,
            "fol_accuracy": float(np.mean(fol_acc)) if fol_acc else 0.0,
            "avg_faithfulness": float(np.mean(faith)) if faith else 0.0,
            "avg_format_score": float(np.mean(fmt)) if fmt else 0.0,
            "avg_latency_s": float(np.mean(lat)) if lat else 0.0,
            "tool_call_rate": tool_calls / max(n, 1),
            "n_examples": n,
        }


# ---------------------------------------------------------------------------
# InternalizationEvaluator
# ---------------------------------------------------------------------------
class InternalizationEvaluator:
    """
    Paired tool / no-tool evaluation for internalization measurement.

    Uses the paired_ablation dataset (built in data/loader.py) where
    each example has:  prompt_with_tool, prompt_without_tool, answer, domain

    Metrics:
      accuracy_with_tools    : acc when model has tool access
      accuracy_without_tools : acc when model has no tool access
      internalization_delta  : with - without  (lower = better)
      consistency_rate       : fraction of examples with same answer both ways
    """

    def __init__(self, model, tokenizer, verifier, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def evaluate(self, paired_dataset: Dataset, max_examples: int = 300) -> dict:
        from ..training.reward_functions import extract_answer

        n = min(max_examples, len(paired_dataset))
        subset = paired_dataset.select(range(n))

        tool_correct, notool_correct, consistent = [], [], []

        self.model.eval()
        logger.info(f"  Internalization eval: {n} paired examples...")

        for i, ex in enumerate(subset):
            c_with = _generate_one(
                self.model, self.tokenizer,
                ex["prompt_with_tool"], self.max_new_tokens
            )
            a_with = extract_answer(c_with)
            v_with = self.verifier.verify(
                ex["domain"], ex["prompt_with_tool"], a_with, ex["answer"]
            )
            tool_correct.append(float(v_with["correct"]))

            c_without = _generate_one(
                self.model, self.tokenizer,
                ex["prompt_without_tool"], self.max_new_tokens
            )
            a_without = extract_answer(c_without)
            v_without = self.verifier.verify(
                ex["domain"], ex["prompt_without_tool"], a_without, ex["answer"]
            )
            notool_correct.append(float(v_without["correct"]))

            consistent.append(
                float(a_with.strip().lower() == a_without.strip().lower())
            )

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                logger.info(
                    f"    [{i+1}/{n}]  "
                    f"notool_acc={np.mean(notool_correct):.3f}  "
                    f"consistency={np.mean(consistent):.3f}"
                )

        self.model.train()
        with_acc = float(np.mean(tool_correct))
        without_acc = float(np.mean(notool_correct))

        return {
            "accuracy_with_tools": with_acc,
            "accuracy_without_tools": without_acc,
            "internalization_delta": with_acc - without_acc,
            "consistency_rate": float(np.mean(consistent)),
            "n_examples": n,
        }


# ---------------------------------------------------------------------------
# Ablation Studies
# ---------------------------------------------------------------------------
def run_ablation_studies(
    with_tools_summary: dict,
    without_tools_summary: dict,
    internalization_results: dict,
    stage_metrics: list,
    curriculum,
    verifier,
) -> dict:
    """
    Run and log all five ablation studies from the thesis.

    Returns a dict with the ablation results (also logged via logger).
    """
    internalization_delta = (
        with_tools_summary["overall_accuracy"]
        - without_tools_summary["overall_accuracy"]
    )
    consistency_rate = internalization_results.get("consistency_rate", 0.0)

    ablations: dict = {}

    # ── Ablation 1: Curriculum stage progression ──────────────────────────
    logger.info("\n🔬 Ablation 1: Curriculum Stage Progression (AdaRFT-style)")
    logger.info("  Stage-wise losses reflect dynamic difficulty adaptation:")
    for m in stage_metrics:
        logger.info(
            f"    Stage {m['stage']}  tool_ratio={m['tool_ratio']}  "
            f"loss={m['loss']:.4f}"
        )
    curr_state = curriculum.get_state()
    logger.info(f"  Final curriculum state: {curr_state}")
    ablations["curriculum"] = {
        "stage_metrics": stage_metrics,
        "final_state": curr_state,
        "note": (
            "Dynamic difficulty scheduling (AdaRFT). "
            "Static baseline requires a separate training run."
        ),
    }

    # ── Ablation 2: Tool reliance ─────────────────────────────────────────
    logger.info("\n🔬 Ablation 2: Tool Reliance")
    logger.info(
        f"  Tool call rate (with-tools eval)  : {with_tools_summary['tool_call_rate']:.4f}"
    )
    logger.info(
        f"  Accuracy with tools               : {with_tools_summary['overall_accuracy']:.4f}"
    )
    logger.info(
        f"  Accuracy without tools            : {without_tools_summary['overall_accuracy']:.4f}"
    )
    logger.info(f"  Internalization delta             : {internalization_delta:.4f}")
    ablations["tool_reliance"] = {
        "tool_call_rate": with_tools_summary["tool_call_rate"],
        "accuracy_with_tools": with_tools_summary["overall_accuracy"],
        "accuracy_without_tools": without_tools_summary["overall_accuracy"],
        "internalization_delta": internalization_delta,
    }

    # ── Ablation 3: Domain-specific performance ───────────────────────────
    logger.info("\n🔬 Ablation 3: Domain-Specific Performance")
    domain_results = {}
    for label, d in [("Math", "math"), ("FOL", "fol")]:
        wk = f"{d}_accuracy"
        logger.info(
            f"  {label}  with={with_tools_summary[wk]:.4f}  "
            f"without={without_tools_summary[wk]:.4f}"
        )
        domain_results[d] = {
            "with_tools": with_tools_summary[wk],
            "without_tools": without_tools_summary[wk],
        }
    ablations["domain"] = domain_results

    # ── Ablation 4: Process faithfulness ─────────────────────────────────
    logger.info("\n🔬 Ablation 4: Process Faithfulness (heuristic PRM)")
    logger.info(
        f"  With tools    : {with_tools_summary['avg_faithfulness']:.4f}"
    )
    logger.info(
        f"  Without tools : {without_tools_summary['avg_faithfulness']:.4f}"
    )
    logger.info(
        "  Thesis note: full neural PRM distillation is future work."
    )
    ablations["faithfulness"] = {
        "with_tools": with_tools_summary["avg_faithfulness"],
        "without_tools": without_tools_summary["avg_faithfulness"],
        "note": "Heuristic PRM; neural PRM distillation is future work.",
    }

    # ── Ablation 5: Efficiency metrics ────────────────────────────────────
    logger.info("\n🔬 Ablation 5: Efficiency Metrics")
    logger.info(
        f"  Avg latency with tools    : {with_tools_summary['avg_latency_s']:.4f}s"
    )
    logger.info(
        f"  Avg latency without tools : {without_tools_summary['avg_latency_s']:.4f}s"
    )
    logger.info(
        f"  Format compliance (w/t)   : {with_tools_summary['avg_format_score']:.4f}"
    )
    logger.info(f"  Consistency rate          : {consistency_rate:.4f}")
    logger.info(f"  Verifier stats            : {verifier.get_stats()}")
    ablations["efficiency"] = {
        "latency_with_tools": with_tools_summary["avg_latency_s"],
        "latency_without_tools": without_tools_summary["avg_latency_s"],
        "format_score": with_tools_summary["avg_format_score"],
        "consistency_rate": consistency_rate,
        "verifier_stats": verifier.get_stats(),
    }

    return ablations


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def run_full_evaluation(
    config,
    model,
    tokenizer,
    verifier,
    eval_verified: Dataset,
    paired_ablation: Dataset,
    stage_metrics: list,
    curriculum,
) -> dict:
    """
    Runs all evaluation phases and saves results to disk.

    Returns the complete results dict (also serialised as JSON).
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    eval_fw = EvaluationFramework(model, tokenizer, verifier)
    intern_eval = InternalizationEvaluator(model, tokenizer, verifier)

    # ── Phase 6: Internalization evaluation ───────────────────────────────
    logger.info("\n📊 Phase 6: Internalization Evaluation")
    internalization_results = intern_eval.evaluate(
        paired_ablation, max_examples=config.internalization_eval_examples
    )
    logger.info(f"  Results: {internalization_results}")

    # ── Phase 7: With-tools vs without-tools ──────────────────────────────
    logger.info("\n📊 Phase 7.1: With-tools evaluation")
    with_tools_summary = eval_fw.evaluate(
        eval_verified,
        max_examples=config.max_eval_examples,
        use_tools=True,
    )

    logger.info("\n📊 Phase 7.2: Without-tools evaluation")
    without_tools_summary = eval_fw.evaluate(
        eval_verified,
        max_examples=config.max_eval_examples,
        use_tools=False,
    )

    internalization_delta = (
        with_tools_summary["overall_accuracy"] - without_tools_summary["overall_accuracy"]
    )

    # ── Print results table ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 60)
    header = f"  {'Metric':<30} {'With Tools':>12} {'No Tools':>12}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for k in [
        "overall_accuracy", "math_accuracy", "fol_accuracy",
        "avg_faithfulness", "avg_format_score", "avg_latency_s", "tool_call_rate",
    ]:
        logger.info(
            f"  {k:<30} {with_tools_summary[k]:>12.4f} {without_tools_summary[k]:>12.4f}"
        )
    logger.info(f"\n  Internalization delta: {internalization_delta:.4f}  (lower = better)")

    # ── Phase 7.3: Ablation studies ───────────────────────────────────────
    logger.info("\n📊 Phase 7.3: Ablation Studies")
    ablations = run_ablation_studies(
        with_tools_summary=with_tools_summary,
        without_tools_summary=without_tools_summary,
        internalization_results=internalization_results,
        stage_metrics=stage_metrics,
        curriculum=curriculum,
        verifier=verifier,
    )

    # ── Assemble full results dict ────────────────────────────────────────
    results = {
        "with_tools": with_tools_summary,
        "without_tools": without_tools_summary,
        "internalization": internalization_results,
        "internalization_delta": internalization_delta,
        "consistency_rate": internalization_results.get("consistency_rate", 0.0),
        "ablations": ablations,
    }

    # Save to JSON
    results_path = str(Path(config.output_dir) / "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nEvaluation results saved → {results_path}")

    # ── Key finding summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)
    if internalization_delta < 0.10:
        logger.info("  ✅ Strong internalization (delta < 0.10)")
    elif internalization_delta < 0.20:
        logger.info("  ⚠️  Moderate internalization (0.10 ≤ delta < 0.20)")
    else:
        logger.info("  ❌ Weak internalization (delta ≥ 0.20) — re-run Stage 2")

    if with_tools_summary["avg_faithfulness"] > 0.50:
        logger.info("  ✅ High faithfulness — traces align with verifier output")
    else:
        logger.info(
            "  ⚠️  Low faithfulness — consider longer thinking traces or PRM distillation"
        )

    return results
