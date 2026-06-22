"""
reporting.py — Experiment Report Builder
=========================================
Assembles and saves the final JSON report and human-readable
summary that form the artefacts section of the thesis appendix.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def build_experiment_report(
    config,
    model,
    tokenizer,
    results: dict,
    stage_metrics: list,
    curriculum,
    verifier,
) -> dict:
    """
    Assemble the full experiment report dict and save to disk.

    Parameters
    ----------
    config        : TrainingConfig
    model         : trained PeftModel
    results       : output of run_full_evaluation()
    stage_metrics : list of per-stage dicts from run_grpo()
    curriculum    : CurriculumScheduler (for final state)
    verifier      : SymbolicVerifier (for cache stats)

    Returns
    -------
    report dict (also saved as JSON + human-readable text)
    """
    from .utils.common import count_parameters

    param_stats = count_parameters(model)
    total_params = param_stats["total"]
    trainable_params = param_stats["trainable"]

    report = {
        "experiment": "CurricSym-SLM-Lite",
        "model": config.model_name,
        "hardware": _get_hardware_info(),
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "parameter_stats": param_stats,
        "results": results,
        "stage_metrics": stage_metrics,
        "curriculum_final_state": curriculum.get_state(),
        "verifier_stats": verifier.get_stats(),
        "ablation_notes": {
            "curriculum": (
                "3-stage AdaRFT tool fading: 1.0→0.5→0.0, dynamic via "
                "CurriculumCallback (reward-driven difficulty adjustment)"
            ),
            "verifier_math": (
                "Z3 numeric oracle — checks pred==gt via UNSAT; "
                "equivalent to float equality but extensible"
            ),
            "verifier_fol": "Z3 propositional consistency check for True/False labels",
            "prm": (
                "Heuristic step-marker PRM (credits thinking length + step words). "
                "Neural PRM distillation is future work."
            ),
            "internalization": (
                "Paired ablation dataset (with-tool vs without-tool prompts). "
                "Delta = accuracy_with - accuracy_without (lower = better)."
            ),
        },
    }

    # ── Save JSON report ───────────────────────────────────────────────────
    report_path = Path(config.output_dir) / "experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report   → {report_path}")

    # ── Save config snapshot ───────────────────────────────────────────────
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(report["config"], f, indent=2, default=str)
    logger.info(f"Config   → {config_path}")

    # ── Print final summary ────────────────────────────────────────────────
    _print_summary(report, total_params, trainable_params, results)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_hardware_info() -> dict:
    info = {}
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
        )
    try:
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


def _print_summary(report: dict, total: int, trainable: int, results: dict) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model              : {report['model']}")
    logger.info(f"Total params       : {total / 1e6:.1f} M")
    logger.info(
        f"Trainable (LoRA)   : {trainable / 1e6:.1f} M  "
        f"({trainable / total * 100:.2f}%)"
    )
    logger.info("")

    wt = results.get("with_tools", {})
    nt = results.get("without_tools", {})
    intern = results.get("internalization", {})

    logger.info(f"Accuracy (w/ tools): {wt.get('overall_accuracy', 0):.4f}")
    logger.info(f"Accuracy (no tools): {nt.get('overall_accuracy', 0):.4f}")
    logger.info(
        f"Internalization Δ  : {results.get('internalization_delta', 0):.4f}  "
        f"(lower = better)"
    )
    logger.info(
        f"Consistency rate   : {results.get('consistency_rate', 0):.4f}"
    )
    logger.info(f"Faithfulness       : {wt.get('avg_faithfulness', 0):.4f}")
    logger.info("=" * 60)
