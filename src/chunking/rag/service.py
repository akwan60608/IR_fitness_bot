"""Text-only retrieval and answer orchestration for the local RAG demo."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from chunking.evals.rag import DEFAULT_RRF_K


class QueryEmbeddings(Protocol):
    """The query half of the persisted index embedding interface."""

    def embed_query(self, text: str) -> list[float]: ...


Answer = Callable[[str, str], str]
Rewrite = Callable[[str], dict[str, str]]
_ROUTES = {"text", "visual", "table"}


class RagService:
    """Answer one question against the text records in one explicit index."""

    def __init__(
        self,
        *,
        document_id: str,
        records: list[dict[str, Any]],
        embeddings: QueryEmbeddings,
        rewrite: Rewrite,
        answer: Answer,
        pipeline_run_id: str | None = None,
    ) -> None:
        self._document_id = document_id
        self._records = [record for record in records if record["record_type"] == "text"]
        if not self._records:
            raise ValueError("Index has no text records for the text-only RAG service.")
        self._embeddings = embeddings
        self._rewrite = rewrite
        self._answer = answer
        self._pipeline_run_id = pipeline_run_id

    def answer(self, question: str) -> dict[str, Any]:
        request_id = str(uuid4())
        clean_question = question.strip() if isinstance(question, str) else ""
        if not clean_question:
            return self._response(
                request_id,
                status="invalid_request",
                error_code="empty_question",
                message="請輸入問題。",
            )
        if len(clean_question) > 500:
            return self._response(
                request_id,
                status="invalid_request",
                error_code="question_too_long",
                message="問題最多 500 個字元。",
            )

        retrieval_query, rewrite = self._rewrite_query(clean_question)
        raw_records = self._retrieve(clean_question)
        records = (
            self._fuse(raw_records, self._retrieve(retrieval_query))
            if rewrite["status"] == "rewritten"
            else raw_records
        )
        sources = [self._source_card(record) for record in records]
        context = "\n\n".join(
            f"[source_id={record['record_id']}]\n{record['text']}" for record in records
        )
        try:
            answer = self._answer(clean_question, context)
        except Exception:
            return self._response(
                request_id,
                status="generation_failed",
                error_code="generation_failed",
                message="無法產生回答，請稍後再試。",
                sources=sources,
                rewrite=rewrite,
            )
        return self._response(
            request_id,
            status="answered",
            answer=answer,
            sources=sources,
            rewrite=rewrite,
        )

    def _rewrite_query(self, question: str) -> tuple[str, dict[str, str]]:
        try:
            result = self._rewrite(question)
            retrieval_query = result.get("retrieval_query") if isinstance(result, dict) else None
            route = result.get("route") if isinstance(result, dict) else None
            if not isinstance(retrieval_query, str) or not retrieval_query.strip() or route not in _ROUTES:
                raise ValueError("invalid rewrite result")
        except Exception:
            return question, {
                "status": "fallback",
                "route": "text",
                "requested_route": "text",
            }
        # shortcut: visual/table requests still retrieve text; add route-specific media and table retrieval in ticket #5.
        return retrieval_query.strip(), {
            "status": "rewritten",
            "route": "text",
            "requested_route": route,
        }

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        query = self._embeddings.embed_query(question)
        # shortcut: linear scan is limited to one local demo index, use a vector database for larger corpora or concurrent traffic.
        return sorted(
            self._records,
            key=lambda record: (-self._cosine(query, record["embedding"]), record["record_id"]),
        )[:3]

    @staticmethod
    def _fuse(
        raw_records: list[dict[str, Any]],
        rewritten_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combined: dict[str, tuple[dict[str, Any], float]] = {}
        for ranked_records in (raw_records, rewritten_records):
            for rank, record in enumerate(ranked_records, start=1):
                record_id = record["record_id"]
                previous = combined.get(record_id)
                combined[record_id] = (
                    record,
                    (previous[1] if previous else 0.0) + 1 / (DEFAULT_RRF_K + rank),
                )
        return [
            record
            for _record_id, (record, _score) in sorted(
                combined.items(),
                key=lambda item: (-item[1][1], item[0]),
            )[:3]
        ]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Query embedding dimension does not match the index.")
        magnitude = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
        return sum(left_value * right_value for left_value, right_value in zip(left, right)) / magnitude if magnitude else 0.0

    def _source_card(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": record["record_id"],
            "document_id": self._document_id,
            "page_numbers": record["page_numbers"],
            "type": record["record_type"],
            "excerpt": record["text"][:240],
        }

    def _response(
        self,
        request_id: str,
        *,
        status: str,
        answer: str | None = None,
        error_code: str | None = None,
        message: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        rewrite: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "request_id": request_id,
            "pipeline_run_id": self._pipeline_run_id,
            "answer": answer,
            "error_code": error_code,
            "message": message,
            "sources": sources or [],
            "rewrite": rewrite or {
                "status": "not_run",
                "route": "text",
                "requested_route": "text",
            },
        }
