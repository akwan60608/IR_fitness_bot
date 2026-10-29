"""Small, separate JSON artifacts for document and pipeline traceability."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parsed_text_records(document: Any) -> list[dict[str, Any]]:
    """Keep parser text separate from cleaned chunks for later diagnosis."""
    return [{
        "doc_item_ref": item.self_ref,
        "label": str(item.label),
        "text": item.text,
        "content_sha256": sha256_text(item.text),
        "source_locations": [provenance.model_dump(mode="json") for provenance in item.prov],
    } for item in document.texts]


def table_records(document: Any, *, document_id: str) -> list[dict[str, Any]]:
    """Export parsed tables separately; regular chunking continues to own retrieval text."""
    records = []
    for index, table in enumerate(document.tables):
        markdown = table.export_to_markdown(document).strip()
        provenance = [item.model_dump(mode="json") for item in table.prov]
        records.append({
            "schema_version": 1,
            "table_id": f"{document_id}:table:{index}:{sha256_text(table.self_ref)[:16]}",
            "document_id": document_id,
            "table_ref": table.self_ref,
            "page_no": provenance[0].get("page_no") if provenance else None,
            "source_locations": provenance,
            "row_count": table.data.num_rows,
            "column_count": table.data.num_cols,
            "markdown": markdown,
            "content_sha256": sha256_text(markdown),
        })
    return records


def write_json(path: Path, value: Any) -> None:
    """Atomically replace one artifact so interrupted runs do not leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        for record in records:
            temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def document_registry(source_path: Path, document_id: str, source_hash: str, page_count: int) -> dict[str, Any]:
    source_stat = source_path.stat()
    return {
        "schema_version": 1,
        "document_id": document_id,
        "source_sha256": source_hash,
        "source": {
            "path": str(source_path.resolve()),
            "filename": source_path.name,
            "mime_type": "application/pdf",
            "byte_size": source_stat.st_size,
            "modified_at": datetime.fromtimestamp(source_stat.st_mtime, UTC).isoformat(),
        },
        "page_count": page_count,
        "registered_at": utc_now(),
        "status": "active",
    }
