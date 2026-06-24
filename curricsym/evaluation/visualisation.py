"""
evaluation/visualisation.py — All thesis result figures.

Figures generated:
  1. Stage losses (curriculum progression)
  2. Tool-fading schedule
  3. Accuracy comparison (with vs without tools)
  4. Process quality metrics
  5. Internalization delta bar chart
  6. OOD robustness gap          ← NEW
  7. Reward curve vs tool_ratio  ← NEW (stage transition visual)
  8. Full 2×4 dashboard

All figures saved as 150 DPI PNG for direct inclusion in LaTeX.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


@_require_mpl
def plot_stage_losses(stage_metrics: list, output_dir: str) -> str:
    stages = [m["stage"] for m in stage_metrics]
    losses = [m["loss"] for m in stage_metrics]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Stage 0\n(Tools=1.0)", "Stage 1\n(Tools=0.5)", "Stage 2\n(Tools=0.0)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stages, losses, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Curriculum Stage", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Loss by Curriculum Stage", fontsize=13, fontweight="bold")
    ax.set_xticks(stages)
    ax.set_xticklabels(labels, fontsize=10)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{loss:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig1_stage_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


@_require_mpl
def plot_tool_fading(stage_metrics: list, output_dir: str) -> str:
    stages = [m["stage"] for m in stage_metrics]
    tr = [m["tool_ratio"] for m in stage_metrics]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(stages, tr, "o-", linewidth=2.5, markersize=10, color="#3498db")
    ax.fill_between(stages, tr, alpha=0.15, color="#3498db")
    ax.set_xlabel("Curriculum Stage", fontsize=12)
    ax.set_ylabel("Tool Ratio", fontsize=12)
    ax.set_title("Tool-Fading Schedule (AdaRFT)", fontsize=13, fontweight="bold")
    ax.set_xticks(stages)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig2_tool_fading.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_accuracy_comparison(with_tools: dict, without_tools: dict, output_dir: str) -> str:
    cats = ["Overall", "Math", "FOL"]
    wacc = [with_tools["overall_accuracy"], with_tools["math_accuracy"], with_tools["fol_accuracy"]]
    nacc = [without_tools["overall_accuracy"], without_tools["math_accuracy"], without_tools["fol_accuracy"]]
    x, w = np.arange(len(cats)), 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w / 2, wacc, w, label="With Tools", color="#2ecc71", edgecolor="white")
    b2 = ax.bar(x + w / 2, nacc, w, label="Without Tools", color="#e74c3c", edgecolor="white")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy: With vs Without Tools", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig3_accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_process_quality(with_tools: dict, consistency_rate: float, output_dir: str) -> str:
    mets = ["Faithfulness", "Format Score", "Tool Efficiency", "Consistency"]
    vals = [with_tools["avg_faithfulness"], with_tools["avg_format_score"],
            with_tools.get("tool_efficiency", 0.0), consistency_rate]
    colors = ["#9b59b6", "#1abc9c", "#f39c12", "#f1c40f"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(mets, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Process Quality Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig4_process_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_internalization_delta(internalization_results: dict, output_dir: str) -> str:
    accs = [internalization_results.get("accuracy_with_tools", 0),
            internalization_results.get("accuracy_without_tools", 0)]
    delta = internalization_results.get("internalization_delta", 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["With Tools", "Without Tools"], accs,
                  color=["#2ecc71", "#e74c3c"], width=0.4, edgecolor="white")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Internalization Analysis  (Δ = {delta:.4f})", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", fontsize=11)
    quality = "✅ Strong" if delta < 0.1 else "⚠️ Moderate" if delta < 0.2 else "❌ Weak"
    ax.annotate(f"Δ = {delta:.4f}\n{quality}", xy=(0.5, 0.82), xycoords="axes fraction",
                ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    path = str(Path(output_dir) / "fig5_internalization_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_ood_robustness(with_tools: dict, ood_results: dict, output_dir: str) -> str:
    """NEW: In-distribution vs OOD accuracy bar chart (GSM-Symbolic main vs p1)."""
    if not ood_results or not ood_results.get("ood_accuracy"):
        return None
    cats = ["In-Dist (main)", "OOD (p1 harder)"]
    vals = [with_tools.get("math_accuracy", 0), ood_results.get("ood_accuracy", 0)]
    gap = ood_results.get("ood_robustness_gap", 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(cats, vals, color=["#3498db", "#e67e22"], width=0.4, edgecolor="white")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"OOD Robustness (Gap = {gap:.4f})", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=11)
    ax.annotate(f"Robustness gap: {gap:.4f}", xy=(0.5, 0.85), xycoords="axes fraction",
                ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf4fb", alpha=0.8))
    plt.tight_layout()
    path = str(Path(output_dir) / "fig6_ood_robustness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def plot_reward_curve(reward_history: list, tool_ratio_history: list,
                      output_dir: str) -> str:
    """NEW: Reward curve overlaid with tool_ratio fade — shows curriculum is working."""
    if not reward_history:
        return None
    steps = list(range(len(reward_history)))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "#3498db"
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Avg Reward (20-step window)", color=color1, fontsize=11)
    # Smooth reward
    window = min(20, len(reward_history))
    smooth = [float(np.mean(reward_history[max(0, i - window):i + 1]))
              for i in range(len(reward_history))]
    ax1.plot(steps, smooth, color=color1, linewidth=2, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.2, 1.2)

    if tool_ratio_history:
        ax2 = ax1.twinx()
        color2 = "#e74c3c"
        ax2.set_ylabel("Tool Ratio", color=color2, fontsize=11)
        ax2.plot(steps[:len(tool_ratio_history)], tool_ratio_history,
                 color=color2, linewidth=2, linestyle="--", label="Tool Ratio")
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-0.05, 1.2)

    ax1.set_title("Reward Curve vs Tool-Fading Schedule", fontsize=13, fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=10)
    plt.tight_layout()
    path = str(Path(output_dir) / "fig7_reward_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


@_require_mpl
def generate_full_dashboard(stage_metrics: list, with_tools: dict,
                             without_tools: dict, internalization_results: dict,
                             consistency_rate: float, output_dir: str,
                             ood_results: dict | None = None) -> str:
    """2×4 master dashboard — primary figure for thesis results section."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("CurricSym-SLM-Lite — Training & Evaluation Results",
                 fontsize=15, fontweight="bold")

    stages = [m["stage"] for m in stage_metrics]
    losses = [m["loss"] for m in stage_metrics]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    # 1) Stage losses
    ax = axes[0, 0]
    ax.bar(stages, losses, color=colors, edgecolor="white")
    ax.set_title("Curriculum Stage Losses")
    ax.set_xticks(stages)
    ax.set_xticklabels(["Early\n(1.0)", "Mid\n(0.5)", "Late\n(0.0)"])
    for s, l, c in zip(stages, losses, colors):
        ax.text(s, l + 0.002, f"{l:.3f}", ha="center", fontsize=9)

    # 2) Tool fading
    ax = axes[0, 1]
    tr = [m["tool_ratio"] for m in stage_metrics]
    ax.plot(stages, tr, "o-", linewidth=2, markersize=8, color="#3498db")
    ax.set_title("Tool-Fading Schedule")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # 3) Internalization delta
    ax = axes[0, 2]
    delta = internalization_results.get("internalization_delta", 0)
    ax.bar(["With Tools", "Without Tools"],
           [internalization_results.get("accuracy_with_tools", 0),
            internalization_results.get("accuracy_without_tools", 0)],
           color=["#2ecc71", "#e74c3c"], width=0.4, edgecolor="white")
    ax.set_title(f"Internalization Δ = {delta:.4f}")
    ax.set_ylim(0, 1.1)

    # 4) OOD robustness (or latency fallback)
    ax = axes[0, 3]
    if ood_results and ood_results.get("ood_accuracy"):
        gap = ood_results.get("ood_robustness_gap", 0)
        ax.bar(["In-Dist", "OOD (p1)"],
               [with_tools.get("math_accuracy", 0), ood_results["ood_accuracy"]],
               color=["#3498db", "#e67e22"], edgecolor="white")
        ax.set_title(f"OOD Robustness Gap={gap:.3f}")
    else:
        ax.bar(["With Tools", "Without Tools"],
               [with_tools["avg_latency_s"], without_tools["avg_latency_s"]],
               color=["#3498db", "#95a5a6"], edgecolor="white")
        ax.set_title("Avg Latency (s)")
    ax.set_ylim(0, 1.1)

    # 5) Domain accuracy
    ax = axes[1, 0]
    cats = ["Overall", "Math", "FOL"]
    wacc = [with_tools["overall_accuracy"], with_tools["math_accuracy"], with_tools["fol_accuracy"]]
    nacc = [without_tools["overall_accuracy"], without_tools["math_accuracy"], without_tools["fol_accuracy"]]
    x, bw = np.arange(len(cats)), 0.35
    ax.bar(x - bw / 2, wacc, bw, label="With Tools", color="#2ecc71", edgecolor="white")
    ax.bar(x + bw / 2, nacc, bw, label="Without Tools", color="#e74c3c", edgecolor="white")
    ax.set_title("Accuracy by Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # 6) Process quality
    ax = axes[1, 1]
    mets = ["Faith.", "Format", "Tool Eff.", "Consist."]
    vals = [with_tools["avg_faithfulness"], with_tools["avg_format_score"],
            with_tools.get("tool_efficiency", 0.0), consistency_rate]
    bars = ax.bar(mets, vals, color=["#9b59b6", "#1abc9c", "#f39c12", "#f1c40f"],
                  edgecolor="white")
    ax.set_title("Process Quality")
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)

    # 7) Latency
    ax = axes[1, 2]
    ax.bar(["With Tools", "Without Tools"],
           [with_tools["avg_latency_s"], without_tools["avg_latency_s"]],
           color=["#3498db", "#95a5a6"], edgecolor="white")
    ax.set_title("Avg Inference Latency (s)")

    # 8) Tool call rate vs efficiency
    ax = axes[1, 3]
    ax.bar(["Tool Call Rate", "Tool Efficiency"],
           [with_tools["tool_call_rate"], with_tools.get("tool_efficiency", 0.0)],
           color=["#e67e22", "#2ecc71"], edgecolor="white")
    ax.set_title("Tool Dependency vs Efficiency")
    ax.set_ylim(0, 1.1)
    for i, val in enumerate([with_tools["tool_call_rate"], with_tools.get("tool_efficiency", 0)]):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    path = str(Path(output_dir) / "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dashboard saved → {path}")
    return path
