"""
evaluation/evaluator.py — Full evaluation suite for thesis metrics.

Metrics computed:
  Primary:
    overall_accuracy, math_accuracy, fol_accuracy
    internalization_delta (core contribution metric)
    consistency_rate

  Process quality:
    avg_faithfulness    (heuristic: trace-verifier alignment)
    avg_format_score    (<thinking>/<answer> compliance)
    tool_call_rate      (fraction using <verify>)
    tool_efficiency     (correct answers WITHOUT tool / total correct)
    avg_latency_s

  OOD robustness:
    ood_accuracy        (GSM-Symbolic p1, harder variant)
    ood_robustness_gap  (in-dist acc - ood acc)

Ablations A1–A5 are all generated from a single training run's data.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared generation helper
# ---------------------------------------------------------------------------
def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    del inputs, out
    return completion


# ---------------------------------------------------------------------------
# EvaluationFramework — primary metrics
# ---------------------------------------------------------------------------
class EvaluationFramework:

    def __init__(self, model, tokenizer, verifier, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def _faithfulness(self, thinking: str, vresult: dict) -> float:
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
        s = 0.0
        if "<thinking>" in completion and "</thinking>" in completion:
            s += 0.3
        if "<answer>" in completion and "</answer>" in completion:
            s += 0.3
        tp, ap = completion.find("<thinking>"), completion.find("<answer>")
        if 0 <= tp < ap:
            s += 0.2
        from ..training.reward_functions import extract_thinking
        if len(extract_thinking(completion)) > 20:
            s += 0.2
        return s

    def evaluate(self, dataset: Dataset, max_examples: int = 150,
                 use_tools: bool = True, tag: str = "") -> dict:
        from ..data.loader import format_prompt_with_tools
        from ..training.reward_functions import extract_answer, extract_thinking

        n = min(max_examples, len(dataset))
        subset = dataset.select(range(n))
        label = f"[{'WITH' if use_tools else 'NO'} TOOLS{' ' + tag if tag else ''}]"

        acc_all, math_acc, fol_acc = [], [], []
        faith, fmt, lat = [], [], []
        tool_calls = 0
        # NEW: tool efficiency — correct answers without <verify>
        correct_no_tool, correct_with_tool = 0, 0

        self.model.eval()
        logger.info(f"  {label} evaluating {n} examples...")

        for i, ex in enumerate(subset):
            prompt = format_prompt_with_tools(
                ex, include_tools=use_tools, tool_ratio=1.0 if use_tools else 0.0)
            t0 = time.time()
            completion = _generate_one(self.model, self.tokenizer, prompt, self.max_new_tokens)
            lat.append(time.time() - t0)

            pred = extract_answer(completion)
            vresult = self.verifier.verify(ex["domain"], ex["prompt"], pred, ex["answer"])
            ok = float(vresult["correct"])
            acc_all.append(ok)
            (math_acc if ex["domain"] == "math" else fol_acc).append(ok)

            faith.append(self._faithfulness(extract_thinking(completion), vresult))
            fmt.append(self._format_score(completion))

            has_verify = "<verify>" in completion
            if has_verify:
                tool_calls += 1
            if ok:
                if has_verify:
                    correct_with_tool += 1
                else:
                    correct_no_tool += 1

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                logger.info(f"    {label} [{i+1}/{n}]  acc={np.mean(acc_all):.3f}")

        self.model.train()
        total_correct = correct_no_tool + correct_with_tool
        return {
            "overall_accuracy": float(np.mean(acc_all)) if acc_all else 0.0,
            "math_accuracy": float(np.mean(math_acc)) if math_acc else 0.0,
            "fol_accuracy": float(np.mean(fol_acc)) if fol_acc else 0.0,
            "avg_faithfulness": float(np.mean(faith)) if faith else 0.0,
            "avg_format_score": float(np.mean(fmt)) if fmt else 0.0,
            "avg_latency_s": float(np.mean(lat)) if lat else 0.0,
            "tool_call_rate": tool_calls / max(n, 1),
            # NEW: fraction of correct answers achieved WITHOUT using the tool
            "tool_efficiency": correct_no_tool / max(total_correct, 1),
            "n_examples": n,
        }


# ---------------------------------------------------------------------------
# OOD Robustness Evaluator
# ---------------------------------------------------------------------------
class OODEvaluator:
    """
    Evaluates on the harder GSM-Symbolic p1 variant (not seen during training).
    Reports ood_accuracy and ood_robustness_gap vs. in-distribution accuracy.
    """

    def __init__(self, model, tokenizer, verifier, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def evaluate(self, ood_dataset: Dataset, in_dist_accuracy: float,
                 max_examples: int = 100) -> dict:
        from ..data.loader import format_prompt_with_tools
        from ..training.reward_functions import extract_answer

        if len(ood_dataset) == 0:
            logger.warning("OOD dataset is empty — skipping OOD eval")
            return {"ood_accuracy": 0.0, "ood_robustness_gap": 0.0, "n_examples": 0}

        n = min(max_examples, len(ood_dataset))
        subset = ood_dataset.select(range(n))
        acc = []

        self.model.eval()
        logger.info(f"  OOD eval: {n} examples (GSM-Symbolic p1)...")

        for i, ex in enumerate(subset):
            prompt = format_prompt_with_tools(ex, include_tools=False, tool_ratio=0.0)
            completion = _generate_one(self.model, self.tokenizer, prompt, self.max_new_tokens)
            pred = extract_answer(completion)
            vr = self.verifier.verify(ex["domain"], ex["prompt"], pred, ex["answer"])
            acc.append(float(vr["correct"]))
            if (i + 1) % 25 == 0:
                torch.cuda.empty_cache()

        self.model.train()
        ood_acc = float(np.mean(acc)) if acc else 0.0
        return {
            "ood_accuracy": ood_acc,
            "ood_robustness_gap": in_dist_accuracy - ood_acc,
            "n_examples": n,
        }


# ---------------------------------------------------------------------------
# InternalizationEvaluator — paired ablation
# ---------------------------------------------------------------------------
class InternalizationEvaluator:
    """
    Paired with-tool / no-tool evaluation.

    Core thesis metric:
      internalization_delta = acc_with_tools - acc_without_tools
      (lower delta = better internalization)
    """

    def __init__(self, model, tokenizer, verifier, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens

    def evaluate(self, paired_dataset: Dataset, max_examples: int = 150) -> dict:
        from ..training.reward_functions import extract_answer

        n = min(max_examples, len(paired_dataset))
        subset = paired_dataset.select(range(n))
        tool_correct, notool_correct, consistent = [], [], []

        self.model.eval()
        logger.info(f"  Internalization eval: {n} paired examples...")

        for i, ex in enumerate(subset):
            c_with = _generate_one(self.model, self.tokenizer,
                                   ex["prompt_with_tool"], self.max_new_tokens)
            a_with = extract_answer(c_with)
            v_with = self.verifier.verify(ex["domain"], ex["prompt_with_tool"],
                                          a_with, ex["answer"])
            tool_correct.append(float(v_with["correct"]))

            c_without = _generate_one(self.model, self.tokenizer,
                                      ex["prompt_without_tool"], self.max_new_tokens)
            a_without = extract_answer(c_without)
            v_without = self.verifier.verify(ex["domain"], ex["prompt_without_tool"],
                                             a_without, ex["answer"])
            notool_correct.append(float(v_without["correct"]))
            consistent.append(float(a_with.strip().lower() == a_without.strip().lower()))

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                logger.info(
                    f"    [{i+1}/{n}]  notool_acc={np.mean(notool_correct):.3f}  "
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
# Ablation Studies (5 structured ablations)
# ---------------------------------------------------------------------------
def run_ablation_studies(with_tools: dict, without_tools: dict,
                         internalization_results: dict, stage_metrics: list,
                         curriculum, verifier, ood_results: dict | None = None) -> dict:
    delta = with_tools["overall_accuracy"] - without_tools["overall_accuracy"]
    consistency = internalization_results.get("consistency_rate", 0.0)
    ablations: dict = {}

    # A1: Curriculum stage progression
    logger.info("\n🔬 Ablation A1: Curriculum Stage Progression")
    for m in stage_metrics:
        logger.info(f"  Stage {m['stage']}  tool={m['tool_ratio']}  loss={m['loss']:.4f}")
    logger.info(f"  Final curriculum: {curriculum.get_state()}")
    ablations["A1_curriculum"] = {
        "stage_metrics": stage_metrics,
        "final_curriculum_state": curriculum.get_state(),
        "note": "AdaRFT dynamic difficulty. Static baseline requires separate run.",
    }

    # A2: Tool reliance analysis
    logger.info("\n🔬 Ablation A2: Tool Reliance")
    logger.info(f"  Tool call rate  : {with_tools['tool_call_rate']:.4f}")
    logger.info(f"  Tool efficiency : {with_tools.get('tool_efficiency', 0):.4f}"
                " (correct answers without tool / total correct)")
    logger.info(f"  Acc w/ tools   : {with_tools['overall_accuracy']:.4f}")
    logger.info(f"  Acc no tools   : {without_tools['overall_accuracy']:.4f}")
    logger.info(f"  Intern. delta  : {delta:.4f}  (lower = better)")
    ablations["A2_tool_reliance"] = {
        "tool_call_rate": with_tools["tool_call_rate"],
        "tool_efficiency": with_tools.get("tool_efficiency", 0.0),
        "accuracy_with_tools": with_tools["overall_accuracy"],
        "accuracy_without_tools": without_tools["overall_accuracy"],
        "internalization_delta": delta,
    }

    # A3: Domain breakdown
    logger.info("\n🔬 Ablation A3: Domain-Specific Performance")
    domain_results = {}
    for label, d in [("Math", "math"), ("FOL", "fol")]:
        wk = f"{d}_accuracy"
        logger.info(f"  {label:5s}  with={with_tools[wk]:.4f}  without={without_tools[wk]:.4f}")
        domain_results[d] = {"with_tools": with_tools[wk], "without_tools": without_tools[wk]}
    ablations["A3_domain"] = domain_results

    # A4: Process faithfulness
    logger.info("\n🔬 Ablation A4: Process Faithfulness (heuristic PRM)")
    logger.info(f"  Faithfulness w/ : {with_tools['avg_faithfulness']:.4f}")
    logger.info(f"  Faithfulness no : {without_tools['avg_faithfulness']:.4f}")
    logger.info("  NOTE: heuristic PRM; neural PRM distillation is future work.")
    ablations["A4_faithfulness"] = {
        "with_tools": with_tools["avg_faithfulness"],
        "without_tools": without_tools["avg_faithfulness"],
    }

    # A5: Efficiency + OOD robustness
    logger.info("\n🔬 Ablation A5: Efficiency & OOD Robustness")
    logger.info(f"  Latency w/ tools  : {with_tools['avg_latency_s']:.4f}s")
    logger.info(f"  Format compliance : {with_tools['avg_format_score']:.4f}")
    logger.info(f"  Consistency rate  : {consistency:.4f}")
    if ood_results:
        logger.info(f"  OOD accuracy (p1) : {ood_results['ood_accuracy']:.4f}")
        logger.info(f"  OOD robustness gap: {ood_results['ood_robustness_gap']:.4f}")
    logger.info(f"  Verifier stats    : {verifier.get_stats()}")
    ablations["A5_efficiency"] = {
        "latency_with_tools": with_tools["avg_latency_s"],
        "format_score": with_tools["avg_format_score"],
        "consistency_rate": consistency,
        "ood": ood_results or {},
        "verifier_stats": verifier.get_stats(),
    }

    return ablations


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def run_full_evaluation(config, model, tokenizer, verifier,
                        eval_verified: Dataset, paired_ablation: Dataset,
                        stage_metrics: list, curriculum,
                        ood_set: Dataset | None = None) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    eval_fw = EvaluationFramework(model, tokenizer, verifier)
    intern_eval = InternalizationEvaluator(model, tokenizer, verifier)
    ood_evaluator = OODEvaluator(model, tokenizer, verifier)

    logger.info("\n📊 Internalization evaluation (paired ablation dataset)...")
    internalization_results = intern_eval.evaluate(
        paired_ablation, max_examples=config.internalization_eval_examples)
    logger.info(f"  Results: {internalization_results}")

    logger.info("\n📊 With-tools evaluation...")
    with_tools = eval_fw.evaluate(eval_verified, config.max_eval_examples, use_tools=True)

    logger.info("\n📊 Without-tools evaluation...")
    without_tools = eval_fw.evaluate(eval_verified, config.max_eval_examples, use_tools=False)

    delta = with_tools["overall_accuracy"] - without_tools["overall_accuracy"]

    # OOD
    ood_results = None
    if ood_set is not None and len(ood_set) > 0 and getattr(config, "run_ood_eval", False):
        logger.info("\n📊 OOD robustness evaluation (GSM-Symbolic p1)...")
        ood_results = ood_evaluator.evaluate(
            ood_set, in_dist_accuracy=with_tools["math_accuracy"],
            max_examples=getattr(config, "ood_eval_examples", 100))
        logger.info(f"  OOD: {ood_results}")

    # Print results table
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 60)
    header = f"  {'Metric':<35} {'With Tools':>10} {'No Tools':>10}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for k in ["overall_accuracy", "math_accuracy", "fol_accuracy",
              "avg_faithfulness", "avg_format_score", "avg_latency_s",
              "tool_call_rate", "tool_efficiency"]:
        logger.info(f"  {k:<35} {with_tools.get(k, 0):>10.4f} {without_tools.get(k, 0):>10.4f}")
    logger.info(f"\n  Internalization delta : {delta:.4f}  (lower = better)")
    if ood_results:
        logger.info(f"  OOD accuracy (p1)    : {ood_results['ood_accuracy']:.4f}")
        logger.info(f"  OOD robustness gap   : {ood_results['ood_robustness_gap']:.4f}")

    ablations = run_ablation_studies(
        with_tools=with_tools, without_tools=without_tools,
        internalization_results=internalization_results,
        stage_metrics=stage_metrics, curriculum=curriculum,
        verifier=verifier, ood_results=ood_results,
    )

    results = {
        "with_tools": with_tools,
        "without_tools": without_tools,
        "internalization": internalization_results,
        "internalization_delta": delta,
        "consistency_rate": internalization_results.get("consistency_rate", 0.0),
        "ood": ood_results or {},
        "ablations": ablations,
    }

    results_path = str(Path(config.output_dir) / "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved → {results_path}")

    # Key finding summary
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)
    label = ("✅ Strong" if delta < 0.10 else "⚠️ Moderate" if delta < 0.20 else "❌ Weak")
    logger.info(f"  {label} internalization (delta={delta:.4f})")
    eff = with_tools.get("tool_efficiency", 0)
    if eff > 0.5:
        logger.info(f"  ✅ High tool efficiency ({eff:.2%} correct without tool)")
    else:
        logger.info(f"  ⚠️ Low tool efficiency ({eff:.2%}) — model still tool-dependent")
    if ood_results and ood_results.get("ood_robustness_gap", 1.0) < 0.10:
        logger.info(f"  ✅ Robust OOD (gap={ood_results['ood_robustness_gap']:.4f})")

    return results
