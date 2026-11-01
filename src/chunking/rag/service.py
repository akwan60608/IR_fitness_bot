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


Answer = Callable[[str, str, tuple[str, ...]], Any]
Rewrite = Callable[[str], dict[str, str]]
_ROUTES = {"text", "visual", "table"}


def rank_records(
    records: list[dict[str, Any]], embeddings: QueryEmbeddings, question: str
) -> list[tuple[dict[str, Any], float]]:
    """Return the local demo's top dense matches and their cosine scores."""
    query = embeddings.embed_query(question)
    # shortcut: linear scan is limited to one local demo index, use a vector database for larger corpora or concurrent traffic.
    return sorted(
        ((record, _cosine(query, record["embedding"])) for record in records),
        key=lambda item: (-item[1], item[0]["record_id"]),
    )[:3]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query embedding dimension does not match the index.")
    magnitude = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(left_value * right_value for left_value, right_value in zip(left, right)) / magnitude if magnitude else 0.0


def fuse_ranked_records(
    raw_records: list[tuple[dict[str, Any], float]],
    rewritten_records: list[tuple[dict[str, Any], float]],
) -> list[dict[str, Any]]:
    """Return the RRF-selected records sent to answer generation."""
    combined: dict[str, tuple[dict[str, Any], float]] = {}
    for ranked_records in (raw_records, rewritten_records):
        for rank, (record, _score) in enumerate(ranked_records, start=1):
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


def gate_score(
    records: list[dict[str, Any]], *ranked_lists: list[tuple[dict[str, Any], float]]
) -> float:
    """Return the strongest dense score among the records kept for generation."""
    return max(
        score
        for record in records
        for ranked in ranked_lists
        for candidate, score in ranked
        if candidate["record_id"] == record["record_id"]
    )


def answer_context(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str]:
    """Create request-local citation aliases without exposing persistent record IDs."""
    aliases = {f"S{index}": record for index, record in enumerate(records, start=1)}
    return aliases, "\n\n".join(
        f"[source_key={alias}]\n{record['text']}" for alias, record in aliases.items()
    )


def render_cited_answer(answer: Any, aliases: dict[str, dict[str, Any]]) -> str | None:
    """Validate structured citation aliases and render the stable source IDs."""
    claims = answer.get("claims") if isinstance(answer, dict) else None
    if not isinstance(claims, list) or not claims:
        return None
    rendered_claims: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            return None
        text = claim.get("text")
        sources = claim.get("sources")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(sources, list)
            or not sources
            or not all(isinstance(source, str) and source in aliases for source in sources)
            or len(set(sources)) != len(sources)
        ):
            return None
        rendered_claims.append(text.strip())
    return "\n".join(rendered_claims)


def rewritten_query(rewrite: Rewrite, question: str) -> tuple[str, str] | None:
    """Return a valid retrieval rewrite, or let callers use their raw query."""
    try:
        result = rewrite(question)
        retrieval_query = result.get("retrieval_query") if isinstance(result, dict) else None
        route = result.get("route") if isinstance(result, dict) else None
        if not isinstance(retrieval_query, str) or not retrieval_query.strip() or route not in _ROUTES:
            return None
    except Exception:
        return None
    return retrieval_query.strip(), route


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
        answerability_threshold: float,
        pipeline_run_id: str | None = None,
    ) -> None:
        self._document_id = document_id
        self._records = [record for record in records if record["record_type"] == "text"]
        if not self._records:
            raise ValueError("Index has no text records for the text-only RAG service.")
        if (
            isinstance(answerability_threshold, bool)
            or not isinstance(answerability_threshold, (int, float))
            or not math.isfinite(answerability_threshold)
            or not -1 <= answerability_threshold <= 1
        ):
            raise ValueError("answerability_threshold must be a finite cosine score from -1 to 1.")
        self._embeddings = embeddings
        self._rewrite = rewrite
        self._answer = answer
        self._answerability_threshold = float(answerability_threshold)
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
        raw_ranked = self._retrieve(clean_question)
        raw_records = [record for record, _score in raw_ranked]
        rewritten_ranked = self._retrieve(retrieval_query) if rewrite["status"] == "rewritten" else []
        records = fuse_ranked_records(raw_ranked, rewritten_ranked) if rewritten_ranked else raw_records
        evidence_score = gate_score(records, raw_ranked, rewritten_ranked)
        if evidence_score < self._answerability_threshold:
            return self._response(
                request_id,
                status="not_answerable",
                error_code="evidence_gate_rejected",
                message="文件中的證據不足以回答這個問題。",
                sources=[self._source_card(record, role="diagnostic") for record in records],
                rewrite=rewrite,
            )
        sources = [self._source_card(record, role="supporting") for record in records]
        aliases, context = answer_context(records)
        try:
            answer = self._answer(clean_question, context, tuple(aliases))
        except Exception:
            return self._response(
                request_id,
                status="generation_failed",
                error_code="generation_failed",
                message="無法產生回答，請稍後再試。",
                sources=[self._source_card(record, role="diagnostic") for record in records],
                rewrite=rewrite,
            )
        rendered_answer = render_cited_answer(answer, aliases)
        if rendered_answer is None:
            return self._response(
                request_id,
                status="generation_failed",
                error_code="citation_validation_failed",
                message="回答的來源引用無效，請稍後再試。",
                sources=[self._source_card(record, role="diagnostic") for record in records],
                rewrite=rewrite,
            )
        return self._response(
            request_id,
            status="answered",
            answer=rendered_answer,
            sources=sources,
            rewrite=rewrite,
        )

    def _rewrite_query(self, question: str) -> tuple[str, dict[str, str]]:
        rewritten = rewritten_query(self._rewrite, question)
        if rewritten is None:
            return question, {
                "status": "fallback",
                "route": "text",
                "requested_route": "text",
            }
        # shortcut: visual/table requests still retrieve text; add route-specific media and table retrieval in ticket #5.
        retrieval_query, requested_route = rewritten
        return retrieval_query, {
            "status": "rewritten",
            "route": "text",
            "requested_route": requested_route,
        }

    def _retrieve(self, question: str) -> list[tuple[dict[str, Any], float]]:
        return rank_records(self._records, self._embeddings, question)

    def _source_card(self, record: dict[str, Any], *, role: str) -> dict[str, Any]:
        return {
            "source_id": record["record_id"],
            "document_id": self._document_id,
            "page_numbers": record["page_numbers"],
            "type": record["record_type"],
            "excerpt": record["text"][:240],
            "role": role,
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
