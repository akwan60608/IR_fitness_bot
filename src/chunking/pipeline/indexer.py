"""Build and load the local, traceable retrieval index."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from chunking.evals.rag import DEFAULT_EMBEDDING_MODEL, E5Embeddings, load_documents
from chunking.pipeline.metadata import sha256_file, sha256_text, utc_now, write_json, write_jsonl


INDEX_SCHEMA_VERSION = 1
INDEX_MANIFEST_NAME = "manifest.json"


class Embeddings(Protocol):
    """The minimal embedding interface used by the persisted index builder."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class LoadedIndex:
    """A validated local index selected explicitly by its directory."""

    manifest: dict[str, Any]
    documents: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"Missing index artifact: {path}.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact: {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    except FileNotFoundError as error:
        raise ValueError(f"Missing index artifact: {path}.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL artifact: {path}.") from error
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Index records must be JSON objects: {path}.")
    return records


def _pipeline_manifest(data_dir: Path, pipeline_run_id: str) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(pipeline_run_id, str)
        or not pipeline_run_id.strip()
        or pipeline_run_id in {".", ".."}
        or "/" in pipeline_run_id
        or "\\" in pipeline_run_id
        or Path(pipeline_run_id).is_absolute()
    ):
        raise ValueError("invalid pipeline_run_id.")
    path = data_dir / "manifests" / f"{pipeline_run_id}.json"
    manifest = _read_json(path)
    if manifest.get("pipeline_run_id") != pipeline_run_id:
        raise ValueError(f"Pipeline manifest at {path} does not match {pipeline_run_id}.")
    if manifest.get("status") != "success":
        raise ValueError(f"Pipeline run {pipeline_run_id} is not successful.")
    return path, manifest


def _safe_document(document: Any, embedding: list[float], index_id: str) -> dict[str, Any]:
    metadata = document.metadata
    record_id = metadata.get("record_id")
    record_type = metadata.get("record_type")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Retrieval document has no record_id.")
    if record_type not in {"text", "media"}:
        raise ValueError(f"Retrieval document {record_id} has an unsupported record_type.")
    text = document.page_content.strip()
    if not text:
        raise ValueError(f"Retrieval document {record_id} has no text.")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"Retrieval document {record_id} has no embedding.")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
        raise ValueError(f"Retrieval document {record_id} has an invalid embedding.")
    source_refs = metadata.get("source_refs", [])
    page_numbers = metadata.get("page_numbers", [])
    if not isinstance(source_refs, list) or not all(isinstance(value, str) for value in source_refs):
        raise ValueError(f"Retrieval document {record_id} has invalid source_refs.")
    if not isinstance(page_numbers, list) or not all(isinstance(value, int) and value > 0 for value in page_numbers):
        raise ValueError(f"Retrieval document {record_id} has invalid page_numbers.")
    picture_ref = metadata.get("picture_ref")
    if picture_ref is not None and not isinstance(picture_ref, str):
        raise ValueError(f"Retrieval document {record_id} has an invalid picture_ref.")
    return {
        "index_id": index_id,
        "record_id": record_id,
        "record_type": record_type,
        "text": text,
        "content_sha256": sha256_text(text),
        "source_refs": sorted(set(source_refs)),
        "page_numbers": sorted(set(page_numbers)),
        "picture_ref": picture_ref,
        "embedding": [float(value) for value in embedding],
    }


def build_index(
    data_dir: Path,
    pipeline_run_id: str,
    index_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embeddings: Embeddings | None = None,
) -> dict[str, Any]:
    """Persist embeddings and UI-safe provenance for one successful pipeline run."""
    if not embedding_model.strip():
        raise ValueError("embedding_model must not be empty.")
    pipeline_manifest_path, pipeline_manifest = _pipeline_manifest(data_dir, pipeline_run_id)
    documents = load_documents(data_dir, pipeline_manifest)
    vectors = (embeddings or E5Embeddings(embedding_model)).embed_documents(
        [document.page_content for document in documents]
    )
    if len(vectors) != len(documents):
        raise ValueError("Embedding provider returned a different number of vectors than documents.")

    index_id = str(uuid4())
    records = [_safe_document(document, vector, index_id) for document, vector in zip(documents, vectors)]
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("Pipeline run contains duplicate retrieval record IDs.")
    dimensions = {len(record["embedding"]) for record in records}
    if len(dimensions) != 1:
        raise ValueError("Embedding provider returned inconsistent vector dimensions.")

    index_dir.mkdir(parents=True, exist_ok=True)
    # shortcut: versioned records accumulate on rebuild; garbage-collect them when index storage becomes material.
    documents_file = f"documents-{index_id}.jsonl"
    write_jsonl(index_dir / documents_file, records)
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "index_id": index_id,
        "created_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "source_manifest_sha256": sha256_file(pipeline_manifest_path),
        "source_sha256": pipeline_manifest.get("source_sha256"),
        "embedding_model": embedding_model,
        "embedding_dimension": dimensions.pop(),
        "document_count": len(records),
        "documents_file": documents_file,
    }
    write_json(index_dir / INDEX_MANIFEST_NAME, manifest)
    return manifest


def _validate_manifest(index_dir: Path) -> dict[str, Any]:
    manifest = _read_json(index_dir / INDEX_MANIFEST_NAME)
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("Index manifest has an unsupported schema_version.")
    for key in ("index_id", "pipeline_run_id", "embedding_model"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Index manifest has no valid {key}.")
    if not isinstance(manifest.get("document_count"), int) or manifest["document_count"] < 1:
        raise ValueError("Index manifest has no valid document_count.")
    if not isinstance(manifest.get("embedding_dimension"), int) or manifest["embedding_dimension"] < 1:
        raise ValueError("Index manifest has no valid embedding_dimension.")
    documents_file = manifest.get("documents_file")
    if not isinstance(documents_file, str) or Path(documents_file).name != documents_file:
        raise ValueError("Index manifest has no safe documents_file.")
    return manifest


def load_index(index_dir: Path) -> LoadedIndex:
    """Load one configured index and reject corrupt or mixed index artifacts."""
    manifest = _validate_manifest(index_dir)
    records = _read_jsonl(index_dir / manifest["documents_file"])
    if len(records) != manifest["document_count"]:
        raise ValueError("Index document count does not match its manifest.")

    seen_record_ids: set[str] = set()
    for record in records:
        if record.get("index_id") != manifest["index_id"]:
            raise ValueError("Index document record does not match its manifest.")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_record_ids:
            raise ValueError("Index has missing or duplicate record IDs.")
        seen_record_ids.add(record_id)
        if record.get("record_type") not in {"text", "media"}:
            raise ValueError(f"Index record {record_id} has an unsupported record_type.")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Index record {record_id} has no text.")
        if record.get("content_sha256") != sha256_text(record["text"]):
            raise ValueError(f"Index record {record_id} has a content hash mismatch.")
        source_refs = record.get("source_refs")
        if not isinstance(source_refs, list) or not all(isinstance(value, str) for value in source_refs):
            raise ValueError(f"Index record {record_id} has invalid source_refs.")
        page_numbers = record.get("page_numbers")
        if not isinstance(page_numbers, list) or not all(isinstance(value, int) and value > 0 for value in page_numbers):
            raise ValueError(f"Index record {record_id} has invalid page_numbers.")
        picture_ref = record.get("picture_ref")
        if picture_ref is not None and not isinstance(picture_ref, str):
            raise ValueError(f"Index record {record_id} has an invalid picture_ref.")
        vector = record.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != manifest["embedding_dimension"]
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
        ):
            raise ValueError(f"Index record {record_id} has an invalid embedding.")
    return LoadedIndex(manifest=manifest, documents=records)
