"""Text-only retrieval and answer orchestration for the local RAG demo."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4


class QueryEmbeddings(Protocol):
    """The query half of the persisted index embedding interface."""

    def embed_query(self, text: str) -> list[float]: ...


Answer = Callable[[str, str], str]


class RagService:
    """Answer one question against the text records in one explicit index."""

    def __init__(
        self,
        *,
        document_id: str,
        records: list[dict[str, Any]],
        embeddings: QueryEmbeddings,
        answer: Answer,
        pipeline_run_id: str | None = None,
    ) -> None:
        self._document_id = document_id
        self._records = [record for record in records if record["record_type"] == "text"]
        if not self._records:
            raise ValueError("Index has no text records for the text-only RAG service.")
        self._embeddings = embeddings
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

        records = self._retrieve(clean_question)
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
            )
        return self._response(request_id, status="answered", answer=answer, sources=sources)

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        query = self._embeddings.embed_query(question)
        # shortcut: linear scan is limited to one local demo index, use a vector database for larger corpora or concurrent traffic.
        return sorted(
            self._records,
            key=lambda record: (-self._cosine(query, record["embedding"]), record["record_id"]),
        )[:3]

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
    ) -> dict[str, Any]:
        return {
            "status": status,
            "request_id": request_id,
            "pipeline_run_id": self._pipeline_run_id,
            "answer": answer,
            "error_code": error_code,
            "message": message,
            "sources": sources or [],
        }
