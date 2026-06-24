"""
curricsym/reporting.py — Generates publication-ready report files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_experiment_report(
    config, model, tokenizer, results: dict,
    stage_metrics: list, curriculum, verifier
) -> dict:
    from .utils import log_environment
    env = log_environment(config)

    report = {
        "experiment_metadata": {
            "model_name": config.model_name,
            "max_seq_length": config.max_seq_length,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "seed": config.seed,
            "environment": env,
        },
        "curriculum_metrics": {
            "stage_metrics": stage_metrics,
            "final_curriculum_state": curriculum.get_state(),
        },
        "evaluation_metrics": results,
        "verifier_statistics": verifier.get_stats(),
    }

    report_path = str(Path(config.output_dir) / "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Master experiment report written to: {report_path}")

    # Generate a quick LaTeX-ready results table text file
    table_path = str(Path(config.output_dir) / "latex_table.txt")
    with open(table_path, "w") as f:
        f.write("% LaTeX table generated for thesis appendix\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        f.write("Metric & With Tools & Without Tools \\\\\n\\hline\n")
        w = results.get("with_tools", {})
        no = results.get("without_tools", {})
        for k in ["overall_accuracy", "math_accuracy", "fol_accuracy",
                  "avg_faithfulness", "avg_format_score", "avg_latency_s", "tool_call_rate"]:
            label = k.replace("_", " ").title()
            f.write(f"{label} & {w.get(k, 0.0):.4f} & {no.get(k, 0.0):.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Internalization Delta & \\multicolumn{{2}}{{c}}{{{results.get('internalization_delta', 0.0):.4f}}} \\\\\n")
        f.write(f"Consistency Rate & \\multicolumn{{2}}{{c}}{{{results.get('consistency_rate', 0.0):.4f}}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\caption{CurricSym-SLM-Lite Evaluation Metrics}\n")
        f.write("\\label{tab:curricsym_results}\n\\end{table}\n")
    logger.info(f"LaTeX-ready table snippet written to: {table_path}")

    return report
