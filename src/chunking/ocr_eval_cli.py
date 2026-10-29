"""Evaluate OCR, media representation, and table artifacts from one pipeline run."""

from __future__ import annotations

import argparse
from pathlib import Path

from chunking.evals.ocr import evaluate_run, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run", required=True, help="Pipeline run ID recorded in data/manifests.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evals/reports"))
    arguments = parser.parse_args()

    report = evaluate_run(arguments.data_dir, arguments.run, arguments.ground_truth)
    output_path = write_report(report, arguments.output_dir, arguments.ground_truth)
    print(f"OCR metrics: {report['metrics']['ocr']}")
    print(f"Visual metrics: {report['metrics']['visual']}")
    print(f"Table metrics: {report['metrics']['table']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
