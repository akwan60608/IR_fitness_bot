"""FastAPI adapter for the local text-only RAG service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel

from chunking.evals.rag import ANSWER_PROMPT, DEFAULT_GROQ_MODEL, E5Embeddings
from chunking.pipeline.indexer import LoadedIndex, load_index
from chunking.rag.service import RagService


class AnswerRequest(BaseModel):
    question: str


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
    index = load_index(Path(index_dir))
    return create_app(_service_from_index(index))


def main() -> None:
    """Start the local service selected by INDEX_DIR."""
    import uvicorn

    uvicorn.run(create_app_from_environment(), host="127.0.0.1", port=8000)


def _service_from_index(index: LoadedIndex) -> RagService:
    model = index.manifest["embedding_model"]
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=0)

    def answer(question: str, context: str) -> str:
        return (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context,
        })

    return RagService(
        document_id=index.manifest["document_id"],
        pipeline_run_id=index.manifest["pipeline_run_id"],
        records=index.documents,
        embeddings=E5Embeddings(model),
        answer=answer,
    )


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
