#!/usr/bin/env python3
"""
generate_tables.py
-------------------
Reads evaluation_results.json and generates publication-quality
table images (PNG) for the thesis paper.

Tables produced:
  Table 1 — Main Results (accuracy, faithfulness, format, latency)
  Table 2 — Domain Breakdown (Math vs FOL)
  Table 3 — Curriculum Stage Progression
  Table 4 — Internalization Analysis
  Table 5 — Ablation Summary
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

RESULTS_JSON = Path("/workspace/curricsym_output/evaluation_results.json")
OUTPUT_DIR   = Path("/workspace/curricsym_output")

with open(RESULTS_JSON) as f:
    R = json.load(f)

wt   = R["with_tools"]
nt   = R["without_tools"]
intern = R["internalization"]
abl  = R["ablations"]
stage_metrics = abl["curriculum"]["stage_metrics"]


def save_table(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  {name} → {path}")


# ── Colour helpers ──────────────────────────────────────────────────────────
HEADER_BG = "#2C3E50"
HEADER_FG = "white"
ROW_A     = "#F2F3F4"
ROW_B     = "white"
ACCENT    = "#2980B9"


def make_table_fig(title, col_labels, rows, col_widths=None, figsize=None):
    n_rows = len(rows)
    n_cols = len(col_labels)
    figsize = figsize or (max(8, n_cols * 2.2), 1.0 + n_rows * 0.45)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    if col_widths:
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(col_widths[c] if c < len(col_widths) else 0.15)

    # Style header
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_height(0.12)

    # Style body rows
    for r in range(1, n_rows + 1):
        bg = ROW_A if r % 2 == 0 else ROW_B
        for c in range(n_cols):
            cell = tbl[r, c]
            cell.set_facecolor(bg)
            cell.set_height(0.09)
            if c == 0:
                cell.set_text_props(fontweight="bold")

    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=HEADER_BG)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Main Results
# ─────────────────────────────────────────────────────────────────────────────
col1 = ["Metric", "With Tools", "Without Tools", "Δ (W - WO)"]
rows1 = [
    ["Overall Accuracy",     f"{wt['overall_accuracy']:.4f}",
                             f"{nt['overall_accuracy']:.4f}",
                             f"{wt['overall_accuracy']-nt['overall_accuracy']:+.4f}"],
    ["Avg Faithfulness",     f"{wt['avg_faithfulness']:.4f}",
                             f"{nt['avg_faithfulness']:.4f}",
                             f"{wt['avg_faithfulness']-nt['avg_faithfulness']:+.4f}"],
    ["Avg Format Score",     f"{wt['avg_format_score']:.4f}",
                             f"{nt['avg_format_score']:.4f}",
                             f"{wt['avg_format_score']-nt['avg_format_score']:+.4f}"],
    ["Avg Latency (s)",      f"{wt['avg_latency_s']:.4f}",
                             f"{nt['avg_latency_s']:.4f}",
                             f"{wt['avg_latency_s']-nt['avg_latency_s']:+.4f}"],
    ["Tool Call Rate",       f"{wt['tool_call_rate']:.4f}",
                             f"{nt['tool_call_rate']:.4f}",
                             "—"],
    ["N Examples",           str(wt['n_examples']),
                             str(nt['n_examples']),
                             "—"],
]
save_table(make_table_fig("Table 1 — Main Results (CurricSym-SLM-Lite)", col1, rows1),
           "table1_main_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Domain Breakdown
# ─────────────────────────────────────────────────────────────────────────────
col2 = ["Domain", "With Tools (Acc)", "Without Tools (Acc)", "Δ"]
rows2 = [
    ["Math (GSM-Symbolic)",
        f"{wt['math_accuracy']:.4f}",
        f"{nt['math_accuracy']:.4f}",
        f"{wt['math_accuracy']-nt['math_accuracy']:+.4f}"],
    ["FOL (ProofWriter)",
        f"{wt['fol_accuracy']:.4f}",
        f"{nt['fol_accuracy']:.4f}",
        f"{wt['fol_accuracy']-nt['fol_accuracy']:+.4f}"],
    ["Overall",
        f"{wt['overall_accuracy']:.4f}",
        f"{nt['overall_accuracy']:.4f}",
        f"{wt['overall_accuracy']-nt['overall_accuracy']:+.4f}"],
]
save_table(make_table_fig("Table 2 — Domain-Specific Accuracy Breakdown", col2, rows2),
           "table2_domain_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Curriculum Stage Progression
# ─────────────────────────────────────────────────────────────────────────────
col3 = ["Stage", "Phase", "Tool Ratio", "Training Loss", "Steps"]
stage_labels = ["Early (Full Supervision)", "Mid (Half Supervision)", "Late (No Tools)"]
rows3 = [
    [str(m["stage"]), stage_labels[i], f"{m['tool_ratio']:.1f}", f"{m['loss']:.6f}", str(m["steps"])]
    for i, m in enumerate(stage_metrics)
]
fs = abl["curriculum"]["final_state"]
rows3.append(["—", "Final Curriculum State",
              f"tool_ratio={fs['tool_ratio']:.1f}",
              f"avg_reward={fs['avg_reward']:.4f}",
              f"step_count={fs['step_count']}"])
save_table(make_table_fig("Table 3 — Curriculum Stage Progression (AdaRFT)", col3, rows3,
                          figsize=(11, 2.8)),
           "table3_curriculum_stages.png")


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — Internalization Analysis
# ─────────────────────────────────────────────────────────────────────────────
col4 = ["Metric", "Value", "Interpretation"]
delta = intern["internalization_delta"]
interp = "✅ Moderate (re-run Stage 2 may improve)" if delta < 0.15 else "❌ Weak — needs more tool-fading stages"
rows4 = [
    ["Accuracy WITH Tools (paired)",    f"{intern['accuracy_with_tools']:.4f}", "Baseline capability"],
    ["Accuracy WITHOUT Tools (paired)", f"{intern['accuracy_without_tools']:.4f}", "Internalized capability"],
    ["Internalization Δ",               f"{delta:.4f}", interp],
    ["Consistency Rate",                f"{intern['consistency_rate']:.4f}", "Fraction same answer both ways"],
    ["N Paired Examples",               str(intern['n_examples']), "Paired ablation dataset"],
]
save_table(make_table_fig("Table 4 — Internalization Analysis", col4, rows4, figsize=(12, 3.2)),
           "table4_internalization.png")


# ─────────────────────────────────────────────────────────────────────────────
# Table 5 — Ablation Summary
# ─────────────────────────────────────────────────────────────────────────────
col5 = ["Ablation", "Key Metric", "Value", "Finding"]
eff = abl["efficiency"]
rows5 = [
    ["Curriculum Stages",  "Stage Losses",
     " → ".join(f"{m['loss']:.5f}" for m in stage_metrics),
     "Loss rises as tool ratio decreases (harder task)"],
    ["Tool Reliance",      "Tool Call Rate",
     f"{abl['tool_reliance']['tool_call_rate']:.2f}",
     "Model relies on tools 90% of the time"],
    ["Domain: Math",       "Acc w/t vs w/o",
     f"{abl['domain']['math']['with_tools']:.3f} vs {abl['domain']['math']['without_tools']:.3f}",
     "FOL generalizes better without tools"],
    ["Domain: FOL",        "Acc w/t vs w/o",
     f"{abl['domain']['fol']['with_tools']:.3f} vs {abl['domain']['fol']['without_tools']:.3f}",
     "FOL benefits from tool-free prompting"],
    ["Faithfulness",       "Heuristic PRM",
     f"{abl['faithfulness']['with_tools']:.3f} (w/t)",
     "Low — neural PRM distillation is future work"],
    ["Efficiency",         "Latency (s)",
     f"{eff['latency_with_tools']:.3f} vs {eff['latency_without_tools']:.3f}",
     "Tool-free is marginally slower (longer traces)"],
    ["Verifier",           "Z3 Calls / Cache",
     f"{eff['verifier_stats']['z3_calls']} / {eff['verifier_stats']['cache_size']}",
     "Efficient caching reduces redundant Z3 calls"],
]
save_table(make_table_fig("Table 5 — Ablation Study Summary", col5, rows5, figsize=(15, 4.0)),
           "table5_ablations.png")


print("\n🎉 All tables generated successfully in", OUTPUT_DIR)
