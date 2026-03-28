"""Repo-root baseline inference entrypoint for automated checks.

This script is intentionally placed at repository root because some
submission validators require `inference.py` at repo root.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from baseline.inference import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline inference on all tasks")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Environment base URL (default: http://localhost:7860)",
    )
    args = parser.parse_args()

    scores: Dict[str, float] = run_baseline(base_url=args.base_url)

    print("Baseline scores:")
    for task_id, score in scores.items():
        print(f"- {task_id}: {score:.3f}")

    # Machine-readable output for automation.
    print(json.dumps({"scores": scores}))


if __name__ == "__main__":
    main()
