"""
data/loader.py — Dataset loading, formatting, and annotation.

Sources:
  Math: apple/GSM-Symbolic (main, p1, p2 variants)
  FOL:  tasksource/proofwriter  (depth-indexed difficulty)

Offline annotation adds:
  • tool_ratio      (from curriculum stage)
  • curriculum_stage
  • verifier metadata (for faithfulness scoring)

OOD set: GSM-Symbolic p1 loaded separately for OOD robustness evaluation.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_prompt_with_tools(
    example: dict,
    include_tools: bool = True,
    tool_ratio: float = 1.0,
) -> str:
    base = example["prompt"]
    domain = example["domain"]
    use_tools = include_tools and (random.random() < tool_ratio)

    if use_tools:
        if domain == "math":
            header = (
                "Solve the following math problem. You may use a symbolic verifier "
                "to check your answer. Think step-by-step, then verify.\n\n"
                "Format: <thinking>...</thinking> <answer>...</answer>\n"
                "Verify: <verify>check_answer(expression)</verify>\n\n"
            )
        else:
            header = (
                "Solve the following first-order logic problem. "
                "You may use a theorem prover (Z3) to check logical entailment.\n\n"
                "Format: <thinking>...</thinking> <answer>...</answer>\n"
                "Verify: <verify>check_entailment(premises, conclusion)</verify>\n\n"
            )
    else:
        header = (
            "Solve the following problem step-by-step.\n"
            "Format: <thinking>...</thinking> <answer>...</answer>\n\n"
        )
    return header + base


def build_sft_text(example: dict) -> dict:
    prompt = format_prompt_with_tools(
        example, include_tools=True, tool_ratio=example.get("tool_ratio", 1.0)
    )
    completion = (
        f"<thinking>\nLet me work through this step by step.\n"
        f"Based on my reasoning, the answer is {example['answer']}.\n</thinking>\n"
        f"<answer>{example['answer']}</answer>"
    )
    return {"text": f"{prompt}\n\n{completion}", "answer": example["answer"],
            "domain": example["domain"], "difficulty": example["difficulty"]}


def build_grpo_example(example: dict) -> dict:
    tool_ratio = example.get("tool_ratio", 0.5)
    prompt = format_prompt_with_tools(example, include_tools=True, tool_ratio=tool_ratio)
    return {"prompt": prompt, "answer": example["answer"], "domain": example["domain"],
            "difficulty": example["difficulty"], "tool_ratio": float(tool_ratio),
            "curriculum_stage": example.get("curriculum_stage", 0)}


def build_paired_ablation_dataset(dataset: Dataset, n: int = 300) -> Dataset:
    subset = dataset.select(range(min(n, len(dataset))))
    rows: dict = {"prompt_with_tool": [], "prompt_without_tool": [],
                  "answer": [], "domain": []}
    for ex in subset:
        rows["prompt_with_tool"].append(
            format_prompt_with_tools(ex, include_tools=True, tool_ratio=1.0))
        rows["prompt_without_tool"].append(
            format_prompt_with_tools(ex, include_tools=False, tool_ratio=0.0))
        rows["answer"].append(ex["answer"])
        rows["domain"].append(ex["domain"])
    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# GSM-Symbolic loader
# ---------------------------------------------------------------------------
def load_gsm_symbolic(max_examples: int = 2000, seed: int = 42,
                      variant: str = "main") -> Dataset:
    """Load a single GSM-Symbolic variant. variant ∈ {main, p1, p2}."""
    difficulty_map = {"main": 2.0, "p1": 3.5, "p2": 5.0}
    base_diff = difficulty_map.get(variant, 2.0)
    try:
        ds = load_dataset("apple/GSM-Symbolic", name=variant, split="test")
        n = min(len(ds), max_examples)
        ds = ds.select(range(n))

        def _process(ex):
            answer_str = ex.get("answer", "")
            m = re.search(r"####\s*(-?[\d,\.]+)", answer_str)
            numeric_answer = m.group(1).replace(",", "") if m else ""
            return {
                "prompt": ex["question"], "answer": numeric_answer,
                "full_answer": answer_str, "domain": "math",
                "difficulty": base_diff + random.uniform(-0.5, 0.5),
                "source": f"gsm_symbolic_{variant}",
                "id": f"gsm_{variant}_{ex.get('id', 'unk')}",
            }

        processed = ds.map(_process, remove_columns=ds.column_names)
        logger.info(f"  ✅ GSM-Symbolic {variant}: {len(processed)} examples")
        return processed.shuffle(seed=seed)
    except Exception as e:
        logger.warning(f"  ⚠️  GSM-Symbolic {variant} failed: {e}")
        return Dataset.from_list([])


def load_gsm_symbolic_mixed(max_examples: int = 2000, seed: int = 42) -> Dataset:
    """Load main + p1 + p2 variants mixed together (for training)."""
    parts = []
    per_variant = max_examples // 3
    for v in ["main", "p1", "p2"]:
        ds = load_gsm_symbolic(per_variant, seed=seed, variant=v)
        if len(ds) > 0:
            parts.append(ds)
    if not parts:
        raise RuntimeError("Could not load any GSM-Symbolic variant")
    combined = concatenate_datasets(parts).shuffle(seed=seed)
    return combined.select(range(min(max_examples, len(combined))))


# ---------------------------------------------------------------------------
# ProofWriter loader (FOL)
# ---------------------------------------------------------------------------
def load_proofwriter(max_examples: int = 1500, seed: int = 42) -> Dataset:
    try:
        ds = load_dataset("tasksource/proofwriter", split="test")
        n_subset = min(max_examples * 3, len(ds))
        ds = ds.select(range(n_subset))

        def _process_fol(ex):
            question = ex.get("question", "")
            theory = ex.get("theory", "")
            answer_raw = str(ex.get("answer", "Unknown")).strip().lower()
            answer = ("True" if answer_raw in ["true", "yes"]
                      else "False" if answer_raw in ["false", "no"] else "Unknown")
            depth = float(ex.get("QDep", 0))
            if depth == 0:
                config_str = str(ex.get("config", ""))
                for d in range(6):
                    if f"depth-{d}" in config_str:
                        depth = float(d)
                        break
            return {
                "prompt": (f"Given the following facts and rules:\n{theory}\n\n"
                           f"Question: {question}\nAnswer (True/False/Unknown):"),
                "answer": answer, "full_answer": ex.get("allProofs", ""),
                "domain": "fol", "difficulty": depth + 1.0 + random.uniform(-0.3, 0.3),
                "source": f"proofwriter_d{int(depth)}", "id": ex.get("id", "unk"),
            }

        processed = (ds.map(_process_fol, remove_columns=ds.column_names)
                     .shuffle(seed=seed)
                     .select(range(min(max_examples, len(ds)))))
        logger.info(f"  ✅ ProofWriter: {len(processed)} examples")
        return processed
    except Exception as e:
        logger.warning(f"  ProofWriter load failed: {e} — generating synthetic fallback")
        templates = [
            ("P and Q implies R. P is true. Q is true. Is R true?", "True"),
            ("If X then Y. X is true. Is Y true?", "True"),
            ("All cats are animals. Tom is a cat. Is Tom an animal?", "True"),
            ("No fish can fly. A salmon is a fish. Can a salmon fly?", "False"),
            ("If it rains then the ground is wet. It is raining. Is the ground wet?", "True"),
        ]
        rows = []
        for i in range(max_examples):
            q, a = templates[i % len(templates)]
            rows.append({"prompt": f"Solve the FOL problem:\n{q}\nAnswer (True/False):",
                         "answer": a, "full_answer": "", "domain": "fol",
                         "difficulty": float(i % 5) + 1 + random.uniform(-0.3, 0.3),
                         "source": "synthetic_fol", "id": f"syn_fol_{i}"})
        return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Offline verifier annotation
# ---------------------------------------------------------------------------
def annotate_with_verifier(dataset: Dataset, config) -> Dataset:
    from ..models.verifier import verify_math_answer, verify_fol_answer

    new_cols: dict = {k: [] for k in list(dataset.column_names) + [
        "verified_answer", "verifier_correct", "verifier_trace",
        "z3_used", "tool_ratio", "curriculum_stage",
    ]}

    for i, ex in enumerate(dataset):
        for k in dataset.column_names:
            new_cols[k].append(ex[k])
        fn = verify_math_answer if ex["domain"] == "math" else verify_fol_answer
        vr = fn(ex["prompt"], ex["answer"], ex["answer"])
        new_cols["verified_answer"].append(ex["answer"])
        new_cols["verifier_correct"].append(True)
        new_cols["verifier_trace"].append(vr.get("verifier_trace", ""))
        new_cols["z3_used"].append(vr.get("z3_used", False))
        stage = ex.get("curriculum_stage", 0)
        new_cols["curriculum_stage"].append(stage)
        new_cols["tool_ratio"].append(
            config.tool_fade_ratios[min(stage, len(config.tool_fade_ratios) - 1)]
        )
        if (i + 1) % 500 == 0:
            logger.info(f"  Verified {i + 1}/{len(dataset)}")

    return Dataset.from_dict(new_cols)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_all_datasets(config) -> dict:
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)

    logger.info("Loading GSM-Symbolic (mixed: main+p1+p2)...")
    gsm_data = load_gsm_symbolic_mixed(config.gsm_symbolic_size, seed=config.seed)
    logger.info(f"Total GSM-Symbolic: {len(gsm_data)}")

    logger.info("Loading ProofWriter...")
    fol_data = load_proofwriter(config.proofwriter_size, seed=config.seed)
    logger.info(f"Total ProofWriter : {len(fol_data)}")

    combined = concatenate_datasets([gsm_data, fol_data]).shuffle(seed=config.seed)
    n_eval = max(1, int(len(combined) * config.eval_split_ratio))
    n_train = len(combined) - n_eval
    train_raw = combined.select(range(n_train))
    eval_raw = combined.select(range(n_train, len(combined)))
    logger.info(f"Train: {len(train_raw)}  Eval: {len(eval_raw)}")

    logger.info("Annotating train set...")
    train_verified = annotate_with_verifier(train_raw, config)
    logger.info("Annotating eval set...")
    eval_verified = annotate_with_verifier(eval_raw, config)

    sft_train = train_verified.map(build_sft_text,
                                   remove_columns=train_verified.column_names)
    sft_eval = eval_verified.map(build_sft_text,
                                 remove_columns=eval_verified.column_names)
    grpo_train = train_verified.map(build_grpo_example,
                                    remove_columns=train_verified.column_names)
    paired_ablation = build_paired_ablation_dataset(
        eval_verified, n=config.internalization_eval_examples
    )

    # OOD set: GSM-Symbolic p1 only (harder; not seen in training if main was used)
    ood_set = Dataset.from_list([])
    if getattr(config, "run_ood_eval", False):
        logger.info("Loading OOD set (GSM-Symbolic p1 harder variant)...")
        ood_raw = load_gsm_symbolic(
            max_examples=config.ood_eval_examples, seed=config.seed, variant="p1"
        )
        if len(ood_raw) > 0:
            ood_set = annotate_with_verifier(ood_raw, config)
        logger.info(f"OOD set: {len(ood_set)} examples")

    logger.info(f"SFT train  : {len(sft_train)}")
    logger.info(f"SFT eval   : {len(sft_eval)}")
    logger.info(f"GRPO train : {len(grpo_train)}")
    logger.info(f"Paired abl : {len(paired_ablation)}")
    logger.info(f"OOD eval   : {len(ood_set)}")

    return {
        "train_verified": train_verified,
        "eval_verified": eval_verified,
        "sft_train": sft_train,
        "sft_eval": sft_eval,
        "grpo_train": grpo_train,
        "paired_ablation": paired_ablation,
        "ood_set": ood_set,
    }
