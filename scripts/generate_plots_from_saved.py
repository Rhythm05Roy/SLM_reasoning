#!/usr/bin/env python3
"""
generate_plots_from_saved.py
-----------------------------
Reads /workspace/curricsym_output/evaluation_results.json and
generates all thesis plots + the unified dashboard.
Run from anywhere:
    python scripts/generate_plots_from_saved.py
"""
import json
import sys
from pathlib import Path

RESULTS_JSON = Path("/workspace/curricsym_output/evaluation_results.json")
OUTPUT_DIR   = "/workspace/curricsym_output"

# ── Locate repo root and import visualisation helpers ──────────────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from curricsym.evaluation.visualisation import (
    plot_stage_losses,
    plot_tool_fading,
    plot_accuracy_comparison,
    plot_process_quality,
    plot_internalization_delta,
    generate_full_dashboard,
)

# ── Load saved results ─────────────────────────────────────────────────────
with open(RESULTS_JSON) as f:
    results = json.load(f)

with_tools            = results["with_tools"]
without_tools         = results["without_tools"]
internalization       = results["internalization"]
consistency_rate      = results.get("consistency_rate", 0.0)
stage_metrics         = results["ablations"]["curriculum"]["stage_metrics"]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Generate individual figures ────────────────────────────────────────────
p1 = plot_stage_losses(stage_metrics, OUTPUT_DIR)
print(f"✅  Fig 1 saved → {p1}")

p2 = plot_tool_fading(stage_metrics, OUTPUT_DIR)
print(f"✅  Fig 2 saved → {p2}")

p3 = plot_accuracy_comparison(with_tools, without_tools, OUTPUT_DIR)
print(f"✅  Fig 3 saved → {p3}")

p4 = plot_process_quality(with_tools, consistency_rate, OUTPUT_DIR)
print(f"✅  Fig 4 saved → {p4}")

p5 = plot_internalization_delta(internalization, OUTPUT_DIR)
print(f"✅  Fig 5 saved → {p5}")

# ── Generate unified dashboard ─────────────────────────────────────────────
p6 = generate_full_dashboard(
    stage_metrics=stage_metrics,
    with_tools=with_tools,
    without_tools=without_tools,
    internalization_results=internalization,
    consistency_rate=consistency_rate,
    output_dir=OUTPUT_DIR,
)
print(f"✅  Dashboard saved → {p6}")
print("\n🎉 All plots generated successfully!")
