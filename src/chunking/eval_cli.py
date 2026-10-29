"""Compare RAG retrieval from two controlled pipeline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from chunking.evals.rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    evaluate_comparison,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evals/questions/half_pdf_30q_clue_answers_review.jsonl"),
    )
    parser.add_argument(
        "--comparison-axis",
        choices=("chunker", "picture_prompt"),
        default="chunker",
        help="chunker compares fixed/P0 with hybrid/P0; picture_prompt compares fixed P0/P1(/P2), or diagnoses one explicit P2 run.",
    )
    parser.add_argument("--fixed-run", help="Exact fixed/P0 run. Supply with --hybrid-run for --comparison-axis chunker.")
    parser.add_argument("--hybrid-run", help="Exact hybrid/P0 run. Supply with --fixed-run for --comparison-axis chunker.")
    parser.add_argument("--p0-run", help="Exact fixed/P0 run. Supply with --p1-run for --comparison-axis picture_prompt.")
    parser.add_argument("--p1-run", help="Exact fixed/P1 run. Supply with --p0-run for --comparison-axis picture_prompt.")
    parser.add_argument("--p2-run", help="Exact fixed/P2 run. Use alone for a P2 diagnostic, or with --p0-run and --p1-run for comparison.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default="dense",
        help="dense preserves the baseline; hybrid combines dense and Chinese BM25 with reciprocal-rank fusion.",
    )
    parser.add_argument(
        "--bm25-ngram",
        type=int,
        default=2,
        help="Chinese character n-gram size for --retriever hybrid (default: 2).",
    )
    parser.add_argument("--answer", action="store_true", help="Also generate non-scoring RAG answers through ChatGroq.")
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evals/reports"))
    arguments = parser.parse_args()

    report = evaluate_comparison(
        arguments.data_dir,
        arguments.questions,
        p0_run=arguments.p0_run,
        p1_run=arguments.p1_run,
        p2_run=arguments.p2_run,
        fixed_run=arguments.fixed_run,
        hybrid_run=arguments.hybrid_run,
        comparison_axis=arguments.comparison_axis,
        embedding_model=arguments.embedding_model,
        top_k=arguments.top_k,
        retriever=arguments.retriever,
        bm25_ngram=arguments.bm25_ngram,
        answer_with_groq=arguments.answer,
        groq_model=arguments.groq_model,
    )
    output_path = write_report(report, arguments.output_dir)
    for variant_name, variant in report["variants"].items():
        metrics = variant["metrics"]
        print(f"{variant_name.upper()} overall: {metrics['overall']}")
        print(f"{variant_name.upper()} media: {metrics['media']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
