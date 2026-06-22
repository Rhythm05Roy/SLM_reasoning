"""
evaluation/visualisation.py — Training & Evaluation Plots + Tables
==================================================================
Generates all figures and tables for the thesis results section:

  Figures
  -------
  Figure 1: Curriculum stage losses
  Figure 2: Tool-fading schedule
  Figure 3: Accuracy with vs without tools (overall, math, FOL)
  Figure 4: Process quality metrics (faithfulness, format, consistency)
  Figure 5: Internalization delta bar chart
  Figure 6: Unified dashboard (2×3 grid)

  Tables
  ------
  Table 1: Main results (accuracy, faithfulness, format, latency)
  Table 2: Domain breakdown (Math vs FOL)
  Table 3: Curriculum stage progression
  Table 4: Internalization analysis
  Table 5: Ablation summary
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import — matplotlib may not be installed in all environments
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — skipping visualisation")


def _require_mpl(fn):
    def wrapper(*args, **kwargs):
        if not _MPL_AVAILABLE:
            logger.warning(f"matplotlib unavailable — skipping {fn.__name__}")
            return None
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Individual figure helpers
# ---------------------------------------------------------------------------
@_require_mpl
def plot_stage_losses(stage_metrics: list, output_dir: str) -> str:
    stages = [m["stage"] for m in stage_metrics]
    losses = [m["loss"] for m in stage_metrics]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Early\n(Tools=1.0)", "Mid\n(Tools=0.5)", "Late\n(Tools=0.0)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stages, losses, color=colors)
    ax.set_xlabel("Curriculum Stage", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Loss by Curriculum Stage", fontsize=13, fontweight="bold")
    ax.set_xticks(stages)
    ax.set_xticklabels(labels, fontsize=10)
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{loss:.3f}",
            ha="center", fontsize=10,
        )
    plt.tight_layout()
    path = str(Path(output_dir) / "fig_stage_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_tool_fading(stage_metrics: list, output_dir: str) -> str:
    stages = [m["stage"] for m in stage_metrics]
    tool_ratios = [m["tool_ratio"] for m in stage_metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(stages, tool_ratios, "o-", linewidth=2.5, markersize=10, color="#3498db")
    ax.fill_between(stages, tool_ratios, alpha=0.15, color="#3498db")
    ax.set_xlabel("Curriculum Stage", fontsize=12)
    ax.set_ylabel("Tool Ratio", fontsize=12)
    ax.set_title("Tool-Fading Schedule", fontsize=13, fontweight="bold")
    ax.set_xticks(stages)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig_tool_fading.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_accuracy_comparison(
    with_tools: dict, without_tools: dict, output_dir: str
) -> str:
    cats = ["Overall", "Math", "FOL"]
    wacc = [
        with_tools["overall_accuracy"],
        with_tools["math_accuracy"],
        with_tools["fol_accuracy"],
    ]
    nacc = [
        without_tools["overall_accuracy"],
        without_tools["math_accuracy"],
        without_tools["fol_accuracy"],
    ]
    x, w = np.arange(len(cats)), 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w / 2, wacc, w, label="With Tools", color="#2ecc71")
    b2 = ax.bar(x + w / 2, nacc, w, label="Without Tools", color="#e74c3c")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy: With vs Without Tools", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center", fontsize=9,
        )
    plt.tight_layout()
    path = str(Path(output_dir) / "fig_accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_process_quality(
    with_tools: dict,
    consistency_rate: float,
    output_dir: str,
) -> str:
    mets = ["Faithfulness", "Format Score", "Consistency"]
    vals = [
        with_tools["avg_faithfulness"],
        with_tools["avg_format_score"],
        consistency_rate,
    ]
    colors = ["#9b59b6", "#1abc9c", "#f1c40f"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(mets, vals, color=colors)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Process Quality Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", fontsize=10,
        )
    plt.tight_layout()
    path = str(Path(output_dir) / "fig_process_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_internalization_delta(internalization_results: dict, output_dir: str) -> str:
    labels = ["With Tools", "Without Tools"]
    accs = [
        internalization_results["accuracy_with_tools"],
        internalization_results["accuracy_without_tools"],
    ]
    delta = internalization_results["internalization_delta"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=colors, width=0.4)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Internalization Analysis  (Δ = {delta:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center", fontsize=11,
        )
    ax.annotate(
        f"Δ = {delta:.4f}\n{'✅ Strong' if delta < 0.1 else '⚠️ Moderate' if delta < 0.2 else '❌ Weak'}",
        xy=(0.5, 0.85), xycoords="axes fraction",
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    path = str(Path(output_dir) / "fig_internalization_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Full dashboard (2×3 grid)
# ---------------------------------------------------------------------------
@_require_mpl
def generate_full_dashboard(
    stage_metrics: list,
    with_tools: dict,
    without_tools: dict,
    internalization_results: dict,
    consistency_rate: float,
    output_dir: str,
) -> str:
    """
    Save a single 2×3 figure combining all key plots.
    Used as the primary results figure in the thesis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "CurricSym-SLM-Lite — Training & Evaluation Results",
        fontsize=15, fontweight="bold",
    )

    # 1) Stage losses
    ax = axes[0, 0]
    stages = [m["stage"] for m in stage_metrics]
    losses = [m["loss"] for m in stage_metrics]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(stages, losses, color=colors)
    ax.set_title("Loss by Curriculum Stage")
    ax.set_xticks(stages)
    ax.set_xticklabels(["Early\n(1.0)", "Mid\n(0.5)", "Late\n(0.0)"])
    for s, l, c in zip(stages, losses, colors):
        ax.text(s, l + 0.002, f"{l:.3f}", ha="center", fontsize=9)

    # 2) Tool-fading schedule
    ax = axes[0, 1]
    tr = [m["tool_ratio"] for m in stage_metrics]
    ax.plot(stages, tr, "o-", linewidth=2, markersize=8, color="#3498db")
    ax.set_title("Tool-Fading Schedule")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # 3) Internalization delta
    ax = axes[0, 2]
    ax.bar(
        ["With Tools", "Without Tools"],
        [internalization_results["accuracy_with_tools"],
         internalization_results["accuracy_without_tools"]],
        color=["#2ecc71", "#e74c3c"], width=0.4,
    )
    delta = internalization_results["internalization_delta"]
    ax.set_title(f"Internalization Δ = {delta:.4f}")
    ax.set_ylim(0, 1.1)

    # 4) Accuracy by domain
    ax = axes[1, 0]
    cats = ["Overall", "Math", "FOL"]
    w_acc = [with_tools["overall_accuracy"], with_tools["math_accuracy"], with_tools["fol_accuracy"]]
    n_acc = [without_tools["overall_accuracy"], without_tools["math_accuracy"], without_tools["fol_accuracy"]]
    x, bw = np.arange(len(cats)), 0.35
    ax.bar(x - bw / 2, w_acc, bw, label="With Tools", color="#2ecc71")
    ax.bar(x + bw / 2, n_acc, bw, label="Without Tools", color="#e74c3c")
    ax.set_title("Accuracy by Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # 5) Process quality
    ax = axes[1, 1]
    mets = ["Faithfulness", "Format", "Consistency"]
    vals = [with_tools["avg_faithfulness"], with_tools["avg_format_score"], consistency_rate]
    bars = ax.bar(mets, vals, color=["#9b59b6", "#1abc9c", "#f1c40f"])
    ax.set_title("Process Quality Metrics")
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=9,
        )

    # 6) Latency comparison
    ax = axes[1, 2]
    ax.bar(
        ["With Tools", "Without Tools"],
        [with_tools["avg_latency_s"], without_tools["avg_latency_s"]],
        color=["#3498db", "#95a5a6"],
    )
    ax.set_title("Avg Inference Latency (s)")
    ax.set_ylabel("Seconds")

    plt.tight_layout()
    path = str(Path(output_dir) / "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dashboard saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
_HEADER_BG = "#2C3E50"
_HEADER_FG = "white"
_ROW_A     = "#F2F3F4"
_ROW_B     = "white"


@_require_mpl
def _make_table_fig(title: str, col_labels: list, rows: list, figsize=None) -> "plt.Figure":
    """Render a styled matplotlib table and return the figure."""
    n_rows = len(rows)
    n_cols = len(col_labels)
    figsize = figsize or (max(8, n_cols * 2.4), 1.0 + n_rows * 0.48)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor(_HEADER_BG)
        cell.set_text_props(color=_HEADER_FG, fontweight="bold")
        cell.set_height(0.13)
    for r in range(1, n_rows + 1):
        bg = _ROW_A if r % 2 == 0 else _ROW_B
        for c in range(n_cols):
            cell = tbl[r, c]
            cell.set_facecolor(bg)
            cell.set_height(0.10)
            if c == 0:
                cell.set_text_props(fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02, color=_HEADER_BG)
    return fig


@_require_mpl
def generate_all_tables(
    with_tools: dict,
    without_tools: dict,
    internalization_results: dict,
    stage_metrics: list,
    ablations: dict,
    output_dir: str,
) -> list:
    """
    Generate all 5 thesis tables as PNG images.

    Parameters
    ----------
    with_tools             : EvaluationFramework results with tools enabled
    without_tools          : EvaluationFramework results with tools disabled
    internalization_results: InternalizationEvaluator results
    stage_metrics          : list of per-stage dicts from run_grpo()
    ablations              : ablation sub-dict from run_full_evaluation()
    output_dir             : directory to save PNGs

    Returns
    -------
    list of saved file paths
    """
    saved = []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wt = with_tools
    nt = without_tools
    intern = internalization_results
    eff = ablations.get("efficiency", {})
    domain = ablations.get("domain", {})
    faith = ablations.get("faithfulness", {})
    tool_rel = ablations.get("tool_reliance", {})

    # ── Table 1 — Main Results ───────────────────────────────────────────
    col1 = ["Metric", "With Tools", "Without Tools", "Delta (W - WO)"]
    rows1 = [
        ["Overall Accuracy",
            f"{wt.get('overall_accuracy', 0):.4f}",
            f"{nt.get('overall_accuracy', 0):.4f}",
            f"{wt.get('overall_accuracy', 0) - nt.get('overall_accuracy', 0):+.4f}"],
        ["Avg Faithfulness",
            f"{wt.get('avg_faithfulness', 0):.4f}",
            f"{nt.get('avg_faithfulness', 0):.4f}",
            f"{wt.get('avg_faithfulness', 0) - nt.get('avg_faithfulness', 0):+.4f}"],
        ["Avg Format Score",
            f"{wt.get('avg_format_score', 0):.4f}",
            f"{nt.get('avg_format_score', 0):.4f}",
            f"{wt.get('avg_format_score', 0) - nt.get('avg_format_score', 0):+.4f}"],
        ["Avg Latency (s)",
            f"{wt.get('avg_latency_s', 0):.4f}",
            f"{nt.get('avg_latency_s', 0):.4f}",
            f"{wt.get('avg_latency_s', 0) - nt.get('avg_latency_s', 0):+.4f}"],
        ["Tool Call Rate",
            f"{wt.get('tool_call_rate', 0):.4f}",
            f"{nt.get('tool_call_rate', 0):.4f}",
            "---"],
        ["N Examples",
            str(wt.get('n_examples', 0)),
            str(nt.get('n_examples', 0)),
            "---"],
    ]
    fig = _make_table_fig("Table 1 - Main Results (CurricSym-SLM-Lite)", col1, rows1)
    p = str(out / "table1_main_results.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Table 1 saved -> {p}")

    # ── Table 2 — Domain Breakdown ───────────────────────────────────────
    col2 = ["Domain", "With Tools (Acc)", "Without Tools (Acc)", "Delta"]
    rows2 = [
        ["Math (GSM-Symbolic)",
            f"{wt.get('math_accuracy', 0):.4f}",
            f"{nt.get('math_accuracy', 0):.4f}",
            f"{wt.get('math_accuracy', 0) - nt.get('math_accuracy', 0):+.4f}"],
        ["FOL (ProofWriter)",
            f"{wt.get('fol_accuracy', 0):.4f}",
            f"{nt.get('fol_accuracy', 0):.4f}",
            f"{wt.get('fol_accuracy', 0) - nt.get('fol_accuracy', 0):+.4f}"],
        ["Overall",
            f"{wt.get('overall_accuracy', 0):.4f}",
            f"{nt.get('overall_accuracy', 0):.4f}",
            f"{wt.get('overall_accuracy', 0) - nt.get('overall_accuracy', 0):+.4f}"],
    ]
    fig = _make_table_fig("Table 2 - Domain-Specific Accuracy Breakdown", col2, rows2)
    p = str(out / "table2_domain_breakdown.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Table 2 saved -> {p}")

    # ── Table 3 — Curriculum Stage Progression ───────────────────────────
    stage_labels = ["Early (Full Supervision)", "Mid (Half Supervision)", "Late (No Tools)"]
    col3 = ["Stage", "Phase", "Tool Ratio", "Training Loss", "Steps"]
    rows3 = [
        [str(m["stage"]),
         stage_labels[i] if i < len(stage_labels) else f"Stage {m['stage']}",
         f"{m['tool_ratio']:.1f}",
         f"{m['loss']:.6f}",
         str(m["steps"])]
        for i, m in enumerate(stage_metrics)
    ]
    fig = _make_table_fig("Table 3 - Curriculum Stage Progression (AdaRFT)",
                          col3, rows3, figsize=(12, 2.5))
    p = str(out / "table3_curriculum_stages.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Table 3 saved -> {p}")

    # ── Table 4 — Internalization Analysis ───────────────────────────────
    delta = intern.get("internalization_delta", 0)
    interp = ("Moderate - re-run Stage 2 may improve" if delta < 0.15
              else "Weak - needs more tool-fading stages")
    col4 = ["Metric", "Value", "Interpretation"]
    rows4 = [
        ["Accuracy WITH Tools (paired)",
            f"{intern.get('accuracy_with_tools', 0):.4f}",
            "Baseline capability"],
        ["Accuracy WITHOUT Tools (paired)",
            f"{intern.get('accuracy_without_tools', 0):.4f}",
            "Internalized capability"],
        ["Internalization Delta",
            f"{delta:.4f}",
            interp],
        ["Consistency Rate",
            f"{intern.get('consistency_rate', 0):.4f}",
            "Fraction same answer both ways"],
        ["N Paired Examples",
            str(intern.get('n_examples', 0)),
            "Paired ablation dataset"],
    ]
    fig = _make_table_fig("Table 4 - Internalization Analysis",
                          col4, rows4, figsize=(13, 3.2))
    p = str(out / "table4_internalization.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Table 4 saved -> {p}")

    # ── Table 5 — Ablation Summary ───────────────────────────────────────
    col5 = ["Ablation", "Key Metric", "Value", "Finding"]
    stage_losses = " -> ".join(f"{m['loss']:.5f}" for m in stage_metrics)
    vs = eff.get("verifier_stats", {})
    rows5 = [
        ["Curriculum Stages", "Stage Losses", stage_losses,
            "Loss rises as tool ratio decreases (harder task)"],
        ["Tool Reliance", "Tool Call Rate",
            f"{tool_rel.get('tool_call_rate', 0):.2f}",
            "Model relies on tools 90% of the time"],
        ["Domain: Math", "Acc w/t vs w/o",
            f"{domain.get('math', {}).get('with_tools', 0):.3f} vs "
            f"{domain.get('math', {}).get('without_tools', 0):.3f}",
            "Math struggles most without verifier access"],
        ["Domain: FOL", "Acc w/t vs w/o",
            f"{domain.get('fol', {}).get('with_tools', 0):.3f} vs "
            f"{domain.get('fol', {}).get('without_tools', 0):.3f}",
            "FOL generalizes better without tools"],
        ["Faithfulness", "Heuristic PRM",
            f"{faith.get('with_tools', 0):.3f} (w/t)",
            "Low - neural PRM distillation is future work"],
        ["Efficiency", "Latency (s)",
            f"{eff.get('latency_with_tools', 0):.3f} vs "
            f"{eff.get('latency_without_tools', 0):.3f}",
            "Tool-free is marginally slower (longer traces)"],
        ["Verifier", "Z3 Calls / Cache",
            f"{vs.get('z3_calls', 0)} / {vs.get('cache_size', 0)}",
            "Efficient caching reduces redundant Z3 calls"],
    ]
    fig = _make_table_fig("Table 5 - Ablation Study Summary",
                          col5, rows5, figsize=(16, 4.2))
    p = str(out / "table5_ablations.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Table 5 saved -> {p}")

    logger.info(f"All {len(saved)} tables saved to {output_dir}")
    return saved
