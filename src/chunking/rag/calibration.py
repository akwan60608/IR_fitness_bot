"""Calibrate the local RAG evidence gate from reviewed questions."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq

from chunking.evals.rag import DEFAULT_GROQ_MODEL, E5Embeddings, load_questions
from chunking.pipeline.indexer import load_index
from chunking.pipeline.metadata import sha256_file
from chunking.rag.api import REWRITE_PROMPT, RewriteResult
from chunking.rag.service import (
    QueryEmbeddings,
    fuse_ranked_records,
    gate_score,
    rank_records,
    rewritten_query,
)


Rewrite = Callable[[str], dict[str, str]]
_REVIEW_STATUSES = {"source_confirmed": True, "document_not_answerable": False}


def calibrate_answerability_threshold(
    *,
    records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    embeddings: QueryEmbeddings,
    rewrite: Rewrite,
) -> dict[str, Any]:
    """Select a reproducible dense-score threshold from reviewed text questions."""
    text_records = [record for record in records if record.get("record_type") == "text"]
    if not text_records:
        raise ValueError("Index has no text records for calibration.")
    scored: list[dict[str, Any]] = []
    for question in questions:
        answerable = _REVIEW_STATUSES.get(question.get("pdf_review", {}).get("review_status"))
        if answerable is None or question.get("target") != "text":
            continue
        raw_ranked = rank_records(text_records, embeddings, question["question"])
        rewritten_ranked = _rewritten_ranked(text_records, embeddings, rewrite, question["question"])
        selected = fuse_ranked_records(raw_ranked, rewritten_ranked) if rewritten_ranked else [
            record for record, _score in raw_ranked
        ]
        scored.append({
            "question_id": question["question_id"],
            "answerable": answerable,
            "dense_score": gate_score(selected, raw_ranked, rewritten_ranked),
        })
    positives = [item for item in scored if item["answerable"]]
    negatives = [item for item in scored if not item["answerable"]]
    if not positives or not negatives:
        raise ValueError("Calibration needs source-confirmed and document-not-answerable text questions.")

    def metrics(threshold: float) -> dict[str, int]:
        return {
            "answerable_total": len(positives),
            "answerable_accepted": sum(item["dense_score"] >= threshold for item in positives),
            "document_not_answerable_total": len(negatives),
            "document_not_answerable_rejected": sum(item["dense_score"] < threshold for item in negatives),
        }

    def selection_key(threshold: float) -> tuple[float, float, float, float]:
        result = metrics(threshold)
        accepted = result["answerable_accepted"] / result["answerable_total"]
        rejected = result["document_not_answerable_rejected"] / result["document_not_answerable_total"]
        return (accepted + rejected, rejected, accepted, -threshold)

    threshold = max({item["dense_score"] for item in scored}, key=selection_key)
    return {
        "answerability_threshold": threshold,
        "metrics": metrics(threshold),
        "questions": scored,
    }


def _rewritten_ranked(
    records: list[dict[str, Any]], embeddings: QueryEmbeddings, rewrite: Rewrite, question: str
) -> list[tuple[dict[str, Any], float]]:
    rewritten = rewritten_query(rewrite, question)
    if rewritten is None:
        return []
    retrieval_query, _route = rewritten
    return rank_records(records, embeddings, retrieval_query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the local RAG answerability gate.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is required to calibrate with the runtime query rewrite.")
    index = load_index(args.index_dir)
    questions = _source_matched_questions(load_questions(args.questions), index.manifest["source_sha256"])
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=0)
    structured_rewriter = llm.with_structured_output(RewriteResult)

    def rewrite(question: str) -> dict[str, str]:
        return (REWRITE_PROMPT | structured_rewriter).invoke({"question": question}).model_dump()

    report = calibrate_answerability_threshold(
        records=index.documents,
        questions=questions,
        embeddings=E5Embeddings(index.manifest["embedding_model"]),
        rewrite=rewrite,
    )
    report.update({
        "index_id": index.manifest["index_id"],
        "pipeline_run_id": index.manifest["pipeline_run_id"],
        "source_sha256": index.manifest["source_sha256"],
        "question_file": str(args.questions),
        "question_file_sha256": sha256_file(args.questions),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _source_matched_questions(questions: list[dict[str, Any]], source_sha256: str) -> list[dict[str, Any]]:
    candidates = [
        question
        for question in questions
        if question.get("target") == "text"
        and question.get("pdf_review", {}).get("review_status") in _REVIEW_STATUSES
    ]
    if not candidates:
        raise ValueError("Question file has no reviewed text questions for calibration.")
    if any(question.get("source", {}).get("source_sha256") != source_sha256 for question in candidates):
        raise ValueError("Calibration questions must match the index source_sha256.")
    return candidates


if __name__ == "__main__":
    main()
