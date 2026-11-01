"""FastAPI adapter for the local text-only RAG service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

from chunking.evals.rag import ANSWER_PROMPT, DEFAULT_GROQ_MODEL, E5Embeddings
from chunking.pipeline.indexer import LoadedIndex, load_index
from chunking.rag.service import RagService


class AnswerRequest(BaseModel):
    question: str


class RewriteResult(BaseModel):
    retrieval_query: str
    route: Literal["text", "visual", "table"]


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the question for retrieval from a fitness document corpus. "
        "Do not answer the question. Select text unless it explicitly asks about an image or table.",
    ),
    ("human", "Question: {question}"),
])


def create_app(service: RagService) -> FastAPI:
    """Expose one preconfigured service without giving HTTP code index access."""
    app = FastAPI()

    @app.post("/answer")
    def answer(request: AnswerRequest) -> StreamingResponse:
        result = service.answer(request.question)
        return StreamingResponse(
            (_sse("result", result), _sse("done", {})),
            media_type="text/event-stream",
        )

    return app


def create_app_from_environment() -> FastAPI:
    """Load the one explicitly selected index before the server starts."""
    index_dir = os.getenv("INDEX_DIR")
    if not index_dir:
        raise RuntimeError("INDEX_DIR must name the local retrieval index directory.")
    answerability_threshold = _answerability_threshold_from_environment()
    index = load_index(Path(index_dir))
    return create_app(_service_from_index(index, answerability_threshold=answerability_threshold))


def _answerability_threshold_from_environment() -> float:
    value = os.getenv("ANSWERABILITY_THRESHOLD")
    try:
        threshold = float(value) if value is not None else None
    except ValueError as error:
        raise RuntimeError("ANSWERABILITY_THRESHOLD must be a cosine score from -1 to 1.") from error
    if threshold is None or not -1 <= threshold <= 1:
        raise RuntimeError("ANSWERABILITY_THRESHOLD must be a cosine score from -1 to 1.")
    return threshold


def main() -> None:
    """Start the local service selected by INDEX_DIR."""
    import uvicorn

    uvicorn.run(create_app_from_environment(), host="127.0.0.1", port=8000)


def _service_from_index(index: LoadedIndex, *, answerability_threshold: float) -> RagService:
    model = index.manifest["embedding_model"]
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=0)

    def answer(question: str, context: str) -> str:
        return (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context,
        })

    structured_rewriter = llm.with_structured_output(RewriteResult)

    def rewrite(question: str) -> dict[str, str]:
        return (REWRITE_PROMPT | structured_rewriter).invoke({"question": question}).model_dump()

    return RagService(
        document_id=index.manifest["document_id"],
        pipeline_run_id=index.manifest["pipeline_run_id"],
        records=index.documents,
        embeddings=E5Embeddings(model),
        rewrite=rewrite,
        answer=answer,
        answerability_threshold=answerability_threshold,
    )


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
