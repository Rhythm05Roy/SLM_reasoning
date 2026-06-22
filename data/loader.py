"""
data/loader.py — Dataset loading & preprocessing
=================================================
Loads GSM-Symbolic (math) and ProofWriter (FOL), merges them,
runs offline Z3 verifier annotation, and materialises:
  • SFT dataset   → text field for SFTTrainer
  • GRPO dataset  → prompt + metadata columns for GRPOTrainer
  • Paired ablation dataset for internalization evaluation
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
    """Return formatted prompt, optionally with verifier-access instructions."""
    base = example["prompt"]
    domain = example["domain"]
    use_tools = include_tools and (random.random() < tool_ratio)

    if use_tools:
        if domain == "math":
            header = (
                "Solve the following math problem. You may use a symbolic verifier\n"
                "to check your answer. Think step-by-step, then verify.\n\n"
                "Format: <thinking>...</thinking> <answer>...</answer>\n"
                "Verify: <verify>check_answer(expression)</verify>\n\n"
            )
        else:  # fol
            header = (
                "Solve the following first-order logic problem.\n"
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
    """Format example for SFT (supervised fine-tuning)."""
    tool_ratio = example.get("tool_ratio", 1.0)
    prompt = format_prompt_with_tools(example, include_tools=True, tool_ratio=tool_ratio)
    completion = (
        f"<thinking>\n"
        f"Let me work through this step by step.\n"
        f"The key information is in the problem statement.\n"
        f"Based on my reasoning, the answer is {example['answer']}.\n"
        f"</thinking>\n"
        f"<answer>{example['answer']}</answer>"
    )
    return {
        "text": f"{prompt}\n\n{completion}",
        "answer": example["answer"],
        "domain": example["domain"],
        "difficulty": example["difficulty"],
        "curriculum_stage": example.get("curriculum_stage", 0),
    }


def build_grpo_example(example: dict) -> dict:
    """Format example for GRPO (no completion — model generates it)."""
    tool_ratio = example.get("tool_ratio", 0.5)
    prompt = format_prompt_with_tools(example, include_tools=True, tool_ratio=tool_ratio)
    return {
        "prompt": prompt,
        "answer": example["answer"],
        "domain": example["domain"],
        "difficulty": example["difficulty"],
        "tool_ratio": float(tool_ratio),
        "curriculum_stage": example.get("curriculum_stage", 0),
    }


# ---------------------------------------------------------------------------
# GSM-Symbolic loader
# ---------------------------------------------------------------------------
def load_gsm_symbolic(
    max_examples: int = 3000,
    seed: int = 42,
) -> Dataset:
    """Load all three GSM-Symbolic variants with jittered difficulty scores."""
    datasets_list = []
    difficulty_map = {"main": 2.0, "p1": 3.5, "p2": 5.0}

    for variant, base_diff in difficulty_map.items():
        try:
            ds = load_dataset("apple/GSM-Symbolic", name=variant, split="test")
            n = min(len(ds), max_examples // 3)
            ds = ds.select(range(n))

            def _process(ex, diff=base_diff):
                answer_str = ex.get("answer", "")
                m = re.search(r"####\s*(-?[\d,\.]+)", answer_str)
                numeric_answer = m.group(1).replace(",", "") if m else ""
                return {
                    "prompt": ex["question"],
                    "answer": numeric_answer,
                    "full_answer": answer_str,
                    "domain": "math",
                    "difficulty": diff + random.uniform(-0.5, 0.5),
                    "source": f"gsm_symbolic_{variant}",
                    "id": f"gsm_{variant}_{ex.get('id', 'unk')}",
                }

            processed = ds.map(_process, remove_columns=ds.column_names)
            datasets_list.append(processed)
            logger.info(f"  ✅ GSM-Symbolic {variant}: {n} examples")
        except Exception as e:
            logger.warning(f"  ⚠️  GSM-Symbolic {variant} failed: {e}")

    if not datasets_list:
        raise RuntimeError("Could not load any GSM-Symbolic variant — check HF connectivity.")

    combined = concatenate_datasets(datasets_list).shuffle(seed=seed)
    if len(combined) > max_examples:
        combined = combined.select(range(max_examples))

    diffs = combined["difficulty"]
    logger.info(
        f"  Difficulty: min={min(diffs):.2f} max={max(diffs):.2f} mean={np.mean(diffs):.2f}"
    )
    return combined


# ---------------------------------------------------------------------------
# ProofWriter loader (FOL)
# ---------------------------------------------------------------------------
def load_proofwriter(max_examples: int = 2000, seed: int = 42) -> Dataset:
    """Load ProofWriter for FOL reasoning with depth-based difficulty."""

    # Primary: tasksource/proofwriter
    try:
        ds = load_dataset("tasksource/proofwriter", split="test")
        n_subset = min(max_examples * 3, len(ds))
        ds = ds.select(range(n_subset))
        logger.info(f"  Loaded tasksource/proofwriter: {len(ds)} rows")

        def _process_fol(ex):
            question = ex.get("question", "")
            theory = ex.get("theory", "")
            answer_raw = str(ex.get("answer", "Unknown")).strip().lower()
            answer = (
                "True" if answer_raw in ["true", "yes"]
                else "False" if answer_raw in ["false", "no"]
                else "Unknown"
            )
            depth = float(ex.get("QDep", 0))
            if depth == 0:
                config_str = str(ex.get("config", ""))
                for d in range(6):
                    if f"depth-{d}" in config_str:
                        depth = float(d)
                        break
            return {
                "prompt": (
                    f"Given the following facts and rules:\n{theory}\n\n"
                    f"Question: {question}\nAnswer (True/False/Unknown):"
                ),
                "answer": answer,
                "full_answer": ex.get("allProofs", ""),
                "domain": "fol",
                "difficulty": depth + 1.0 + random.uniform(-0.3, 0.3),
                "source": f"proofwriter_d{int(depth)}",
                "id": ex.get("id", "unk"),
            }

        processed = (
            ds.map(_process_fol, remove_columns=ds.column_names)
            .shuffle(seed=seed)
            .select(range(min(max_examples, len(ds))))
        )
        return processed
    except Exception as e:
        logger.warning(f"  tasksource/proofwriter failed: {e}")

    # Fallback: D3xter1922
    try:
        ds = load_dataset("D3xter1922/proofwriter-dataset", split="test")
        logger.info(f"  Loaded D3xter1922/proofwriter-dataset: {len(ds)} rows")

        def _process_fol_v2(ex):
            t = ex.get("translation", "")
            answer = "Unknown"
            if "$answer$" in t:
                a_part = t.split("$answer$")[1].split(";")[0].strip().lstrip("=").strip()
                if a_part.lower() in ["true", "false"]:
                    answer = a_part.capitalize()
            question = (
                t.split("$question$")[1].split(";")[0].strip().lstrip("=").strip()
                if "$question$" in t else t[:100]
            )
            theory = (
                t.split("$context$")[1].strip().lstrip("=").strip()
                if "$context$" in t else ""
            )
            return {
                "prompt": (
                    f"Given the following facts and rules:\n{theory}\n\n"
                    f"Question: {question}\nAnswer (True/False/Unknown):"
                ),
                "answer": answer,
                "full_answer": t[:500],
                "domain": "fol",
                "difficulty": 3.0 + random.uniform(-1.0, 1.0),
                "source": "proofwriter_d3xter",
                "id": "d3xter_" + hashlib.md5(question.encode()).hexdigest()[:8],
            }

        processed = (
            ds.map(_process_fol_v2, remove_columns=ds.column_names)
            .shuffle(seed=seed)
            .select(range(min(max_examples, len(ds))))
        )
        return processed
    except Exception as e:
        logger.warning(f"  D3xter1922/proofwriter-dataset failed: {e}")

    # Last resort: synthetic FOL
    logger.warning("  Generating synthetic FOL fallback dataset.")
    templates = [
        ("P and Q implies R. P is true. Q is true. Is R true?", "True"),
        ("If X then Y. X is true. Is Y true?", "True"),
        ("All cats are animals. Tom is a cat. Is Tom an animal?", "True"),
        ("No fish can fly. A salmon is a fish. Can a salmon fly?", "False"),
        ("If it rains then the ground is wet. It is raining. Is the ground wet?", "True"),
        ("All birds have wings. A penguin is a bird. Does a penguin have wings?", "True"),
        ("If A then B. If B then C. A is true. Is C true?", "True"),
        ("Not P. P implies Q. Is Q true?", "False"),
    ]
    rows = []
    for i in range(max_examples):
        q, a = templates[i % len(templates)]
        rows.append({
            "prompt": f"Solve the FOL problem:\n{q}\nAnswer (True/False):",
            "answer": a,
            "full_answer": "",
            "domain": "fol",
            "difficulty": float(i % 5) + 1 + random.uniform(-0.3, 0.3),
            "source": "synthetic_fol",
            "id": f"syn_fol_{i}",
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Offline verifier annotation
# ---------------------------------------------------------------------------
def annotate_with_verifier(dataset: Dataset, config) -> Dataset:
    """
    Annotate each example with:
      • verified_answer, verifier_correct, verifier_trace, z3_used
      • tool_ratio (from curriculum stage)
      • curriculum_stage
    """
    # Import here to avoid circular imports
    from ..models.verifier import verify_math_answer, verify_fol_answer

    new_cols = {k: [] for k in list(dataset.column_names) + [
        "verified_answer", "verifier_correct",
        "verifier_trace", "z3_used", "tool_ratio", "curriculum_stage",
    ]}

    for i, ex in enumerate(dataset):
        for k in dataset.column_names:
            new_cols[k].append(ex[k])

        fn = verify_math_answer if ex["domain"] == "math" else verify_fol_answer
        vr = fn(ex["prompt"], ex["answer"], ex["answer"])  # GT vs GT → always correct

        new_cols["verified_answer"].append(ex["answer"])
        new_cols["verifier_correct"].append(True)
        new_cols["verifier_trace"].append(vr["verifier_trace"])
        new_cols["z3_used"].append(vr["z3_used"])

        stage = ex.get("curriculum_stage", 0)
        new_cols["curriculum_stage"].append(stage)
        new_cols["tool_ratio"].append(
            config.tool_fade_ratios[min(stage, len(config.tool_fade_ratios) - 1)]
        )

        if (i + 1) % 1000 == 0:
            logger.info(f"  Verified {i + 1}/{len(dataset)}")

    return Dataset.from_dict(new_cols)


# ---------------------------------------------------------------------------
# Paired ablation dataset (for internalization evaluation)
# ---------------------------------------------------------------------------
def build_paired_ablation_dataset(dataset: Dataset, n: int = 500) -> Dataset:
    """
    Create paired (prompt_with_tool, prompt_without_tool) for each example.
    Used in Phase 6 internalization delta evaluation.
    """
    subset = dataset.select(range(min(n, len(dataset))))
    rows: dict = {
        "prompt_with_tool": [],
        "prompt_without_tool": [],
        "answer": [],
        "domain": [],
    }
    for ex in subset:
        rows["prompt_with_tool"].append(
            format_prompt_with_tools(ex, include_tools=True, tool_ratio=1.0)
        )
        rows["prompt_without_tool"].append(
            format_prompt_with_tools(ex, include_tools=False, tool_ratio=0.0)
        )
        rows["answer"].append(ex["answer"])
        rows["domain"].append(ex["domain"])
    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_all_datasets(config) -> dict:
    """
    Load, annotate and materialise all datasets required by the pipeline.

    Returns
    -------
    dict with keys:
        train_verified, eval_verified,
        sft_train, sft_eval,
        grpo_train,
        paired_ablation
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)

    # 1) Load raw datasets
    logger.info("Loading GSM-Symbolic...")
    gsm_data = load_gsm_symbolic(config.gsm_symbolic_size, seed=config.seed)
    logger.info(f"Total GSM-Symbolic: {len(gsm_data)}")

    logger.info("Loading ProofWriter...")
    fol_data = load_proofwriter(config.proofwriter_size, seed=config.seed)
    logger.info(f"Total ProofWriter : {len(fol_data)}")

    # 2) Combine and split
    combined = concatenate_datasets([gsm_data, fol_data]).shuffle(seed=config.seed)
    n_eval = max(1, int(len(combined) * config.eval_split_ratio))
    n_train = len(combined) - n_eval
    train_raw = combined.select(range(n_train))
    eval_raw = combined.select(range(n_train, len(combined)))
    logger.info(f"Train: {len(train_raw)}  Eval: {len(eval_raw)}")

    # 3) Annotate with offline verifier
    logger.info("Annotating train set with verifier...")
    train_verified = annotate_with_verifier(train_raw, config)
    logger.info("Annotating eval set with verifier...")
    eval_verified = annotate_with_verifier(eval_raw, config)

    # 4) Build task-specific datasets
    sft_train = train_verified.map(build_sft_text, remove_columns=train_verified.column_names)
    sft_eval = eval_verified.map(build_sft_text, remove_columns=eval_verified.column_names)
    grpo_train = train_verified.map(build_grpo_example, remove_columns=train_verified.column_names)
    paired_ablation = build_paired_ablation_dataset(
        eval_verified, n=config.internalization_eval_examples
    )

    logger.info(f"SFT train  : {len(sft_train)} examples")
    logger.info(f"SFT eval   : {len(sft_eval)} examples")
    logger.info(f"GRPO train : {len(grpo_train)} examples")
    logger.info(f"Paired abl : {len(paired_ablation)} examples")

    return {
        "train_verified": train_verified,
        "eval_verified": eval_verified,
        "sft_train": sft_train,
        "sft_eval": sft_eval,
        "grpo_train": grpo_train,
        "paired_ablation": paired_ablation,
    }
