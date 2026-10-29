from __future__ import annotations

from typing import Literal

from docling.chunking import HierarchicalChunker, HybridChunker


from chunking.pipeline.metadata import sha256_text


CHUNKER_NAME = "HybridChunker"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ChunkerStrategy = Literal["fixed", "hybrid"]
FIXED_CHUNKER_NAME = "FixedSizeChunker"


def _source_locations(doc_items: list) -> list[dict]:
    locations = []
    for item in doc_items:
        for provenance in item.prov:
            location = provenance.model_dump(mode="json")
            location["doc_item_ref"] = item.self_ref
            location["label"] = str(item.label)
            locations.append(location)
    return locations


def chunker_name(strategy: ChunkerStrategy) -> str:
    return CHUNKER_NAME if strategy == "hybrid" else FIXED_CHUNKER_NAME


def _fixed_segments(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split only an oversized source item; ordinary boundaries stay fixed-size."""
    if tokenizer.count_tokens(text=text) <= max_tokens:
        return [text]
    token_ids = tokenizer.get_tokenizer()(text, add_special_tokens=False)["input_ids"]
    return [
        tokenizer.get_tokenizer().decode(token_ids[start : start + max_tokens], skip_special_tokens=True).strip()
        for start in range(0, len(token_ids), max_tokens)
    ]


def _fixed_size_chunks(doc, tokenizer) -> list[tuple[str, list, list[str]]]:
    """Pack Docling's reading-order items without heading-aware peer merging."""
    max_tokens = tokenizer.get_max_tokens()
    chunks: list[tuple[str, list, list[str]]] = []
    text_parts: list[str] = []
    doc_items: list = []
    headings: list[str] = []

    def flush() -> None:
        nonlocal text_parts, doc_items, headings
        if text_parts:
            chunks.append(("\n".join(text_parts), doc_items, headings))
        text_parts, doc_items, headings = [], [], []

    for element in HierarchicalChunker().chunk(dl_doc=doc):
        for segment in _fixed_segments(element.text.strip(), tokenizer, max_tokens):
            if not segment:
                continue
            candidate = "\n".join([*text_parts, segment])
            if text_parts and tokenizer.count_tokens(text=candidate) > max_tokens:
                flush()
            text_parts.append(segment)
            doc_items.extend(element.meta.doc_items)
            for heading in element.meta.headings or []:
                if heading not in headings:
                    headings.append(heading)
    flush()
    return chunks


def _record(
    *,
    text: str,
    raw_text: str,
    doc_items: list,
    headings: list[str],
    document_id: str,
    pipeline_run_id: str,
    chunk_index: int,
    tokenizer,
    media_records: list[dict],
) -> dict:
    content_hash = sha256_text(raw_text)
    item_refs = {item.self_ref for item in doc_items}
    return {
        "schema_version": 1,
        "chunk_id": f"{document_id}:{chunk_index}:{content_hash[:16]}",
        "document_id": document_id,
        "pipeline_run_id": pipeline_run_id,
        "chunk_index": chunk_index,
        "text": text,
        "raw_text": raw_text,
        "content_sha256": content_hash,
        "token_count": tokenizer.count_tokens(text=raw_text),
        "headings": headings,
        "source_locations": _source_locations(doc_items),
        "media_ids": sorted(
            record["media_id"]
            for record in media_records
            if item_refs.intersection([*record["caption_refs"], record.get("picture_ref")])
        ),
    }


def chunk_document(
    doc,
    *,
    document_id: str,
    pipeline_run_id: str,
    media_records: list[dict] | None = None,
    strategy: ChunkerStrategy = "hybrid",
) -> list[dict]:
    """Return compact chunk records linked to separate document/run artifacts."""
    if strategy not in {"fixed", "hybrid"}:
        raise ValueError(f"Unsupported chunker strategy: {strategy}")
    chunker = HybridChunker()
    chunks = []
    media_records = media_records or []

    if strategy == "fixed":
        for i, (raw_text, doc_items, headings) in enumerate(_fixed_size_chunks(doc, chunker.tokenizer)):
            chunks.append(_record(
                text=raw_text,
                raw_text=raw_text,
                doc_items=doc_items,
                headings=headings,
                document_id=document_id,
                pipeline_run_id=pipeline_run_id,
                chunk_index=i,
                tokenizer=chunker.tokenizer,
                media_records=media_records,
            ))
        return chunks

    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        chunks.append(_record(
            text=chunker.contextualize(chunk),
            raw_text=chunk.text,
            doc_items=chunk.meta.doc_items,
            headings=chunk.meta.headings or [],
            document_id=document_id,
            pipeline_run_id=pipeline_run_id,
            chunk_index=i,
            tokenizer=chunker.tokenizer,
            media_records=media_records,
        ))

    return chunks
