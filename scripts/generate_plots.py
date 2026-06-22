#!/usr/bin/env python3
"""
scripts/generate_plots.py — Re-generate all figures from saved JSON results
===========================================================================
Useful when you want to tweak plot aesthetics without re-running training.

Usage
-----
  python scripts/generate_plots.py \
    --results_json /workspace/curricsym_output/evaluation_results.json \
    --stage_metrics_json /workspace/curricsym_output/grpo_stage_metrics.json \
    --output_dir   /workspace/curricsym_output/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curricsym.evaluation import (
    generate_full_dashboard,
    plot_stage_losses,
    plot_tool_fading,
    plot_accuracy_comparison,
    plot_process_quality,
    plot_internalization_delta,
)
from curricsym.utils import get_logger

logger = get_logger("generate_plots")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-generate thesis figures from saved JSON")
    p.add_argument("--results_json",      required=True,
                   help="Path to evaluation_results.json")
    p.add_argument("--stage_metrics_json", required=True,
                   help="Path to grpo_stage_metrics.json")
    p.add_argument("--output_dir",        default="./figures",
                   help="Where to save generated figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.results_json) as f:
        results = json.load(f)
    with open(args.stage_metrics_json) as f:
        stage_metrics = json.load(f)

    with_tools = results.get("with_tools", {})
    without_tools = results.get("without_tools", {})
    internalization = results.get("internalization", {})
    consistency_rate = results.get("consistency_rate", 0.0)

    logger.info(f"Generating figures → {args.output_dir}")

    paths = [
        plot_stage_losses(stage_metrics, args.output_dir),
        plot_tool_fading(stage_metrics, args.output_dir),
        plot_accuracy_comparison(with_tools, without_tools, args.output_dir),
        plot_process_quality(with_tools, consistency_rate, args.output_dir),
        plot_internalization_delta(internalization, args.output_dir),
        generate_full_dashboard(
            stage_metrics=stage_metrics,
            with_tools=with_tools,
            without_tools=without_tools,
            internalization_results=internalization,
            consistency_rate=consistency_rate,
            output_dir=args.output_dir,
        ),
    ]

    logger.info("Generated files:")
    for p in paths:
        if p:
            logger.info(f"  {p}")
    logger.info("✅ Done")


if __name__ == "__main__":
    main()
