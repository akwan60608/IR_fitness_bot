"""Export Docling pictures and apply only deterministic retention rules."""

from __future__ import annotations

from pathlib import Path

from chunking.pipeline.cleaner import clean_text
from chunking.pipeline.metadata import sha256_file, sha256_text
from chunking.pipeline.picture_enrichment import p2_picture_kind


NOISE_CLASSES = {"bar_code", "icon", "logo", "page_thumbnail", "qr_code", "signature", "stamp"}
PICTURE_CONTEXT_STRATEGY = "same_page_figure_context_v2"
PICTURE_CONTEXT_MAX_CHARS = 1200
PICTURE_CONTEXT_MAX_ITEMS = 5


def _is_no_retrieval_description(text: str | None) -> bool:
    return text is not None and text.strip() == "NO_RETRIEVAL"


def _description_text(picture) -> str | None:
    return getattr(getattr(getattr(picture, "meta", None), "description", None), "text", None)


def _picture_area_fraction(document, picture) -> float | None:
    if not picture.prov:
        return None
    provenance = picture.prov[0]
    page_no = getattr(provenance, "page_no", None)
    page = getattr(document, "pages", {}).get(page_no)
    if page is None:
        return None
    page_area = page.size.width * page.size.height
    return provenance.bbox.area() / page_area if page_area else None


def picture_description_stats(document, *, area_threshold: float) -> dict[str, int]:
    """Report why Docling picture descriptions were or were not persisted."""
    counts = {
        "generated": 0,
        "skipped_area": 0,
        "skipped_classification": 0,
        "not_generated_other": 0,
    }
    for picture in document.pictures:
        if _description_text(picture):
            counts["generated"] += 1
        elif (area := _picture_area_fraction(document, picture)) is not None and area < area_threshold:
            counts["skipped_area"] += 1
        elif _has_noise_classification(picture):
            counts["skipped_classification"] += 1
        else:
            counts["not_generated_other"] += 1
    return counts


def picture_description_record_stats(records: list[dict]) -> dict[str, int]:
    """Count P2 outcomes from its media records, not Docling's skipped VLM stage."""
    counts = {
        "generated": 0,
        "skipped_decorative": 0,
        "skipped_no_image": 0,
        "failed": 0,
        "empty": 0,
    }
    for record in records:
        status = (record.get("p2") or {}).get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _class_name(picture) -> tuple[str | None, float | None]:
    classification = getattr(getattr(picture, "meta", None), "classification", None)
    if classification is None:
        return None, None
    prediction = classification.get_main_prediction()
    class_name = prediction.class_name
    return getattr(class_name, "value", str(class_name)), prediction.confidence


def _has_noise_classification(picture) -> bool:
    classification = getattr(getattr(picture, "meta", None), "classification", None)
    predictions = getattr(classification, "predictions", ())
    if predictions:
        return any(
            getattr(prediction.class_name, "value", str(prediction.class_name)) in NOISE_CLASSES
            for prediction in predictions
        )
    return _class_name(picture)[0] in NOISE_CLASSES


def _decision(
    class_name: str | None,
    page_no: int | None,
    description: str | None,
    *,
    instruction_override: bool = False,
) -> dict:
    if _is_no_retrieval_description(description):
        role = "decorative"
    elif class_name == "icon" and instruction_override:
        role = "instructional_figure"
    elif class_name in {"qr_code", "bar_code"}:
        role = "qr_link"
    elif class_name in NOISE_CLASSES:
        role = "decorative"
    elif class_name == "full_page_image" and page_no == 1:
        role = "cover"
    else:
        role = "unknown"
    return {
        "role": role,
        "retrieval_eligible": role in {"unknown", "instructional_figure"},
        "decision_model": "docling_classifier_rules+instruction_override" if instruction_override else "docling_classifier_rules",
        "needs_review": role == "unknown",
    }


def _save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(path)


def _provenance_data(item) -> list[dict]:
    return [provenance.model_dump(mode="json") for provenance in getattr(item, "prov", [])]


def _center(provenance: dict) -> tuple[float, float] | None:
    bbox = provenance.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        return ((float(bbox["l"]) + float(bbox["r"])) / 2, (float(bbox["t"]) + float(bbox["b"])) / 2)
    except (KeyError, TypeError, ValueError):
        return None


def _contains(provenance: dict, point: tuple[float, float] | None) -> bool:
    bbox = provenance.get("bbox")
    if point is None or not isinstance(bbox, dict):
        return False
    try:
        return (
            min(float(bbox["l"]), float(bbox["r"])) <= point[0] <= max(float(bbox["l"]), float(bbox["r"]))
            and min(float(bbox["b"]), float(bbox["t"])) <= point[1] <= max(float(bbox["b"]), float(bbox["t"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _picture_context(document, picture, *, max_chars: int, class_name: str | None) -> dict | None:
    """Keep bounded, traceable nearby text with a P2 picture retrieval record."""
    picture_provenance = _provenance_data(picture)
    if not picture_provenance:
        return None
    page_no = picture_provenance[0].get("page_no")
    if not isinstance(page_no, int):
        return None
    picture_center = _center(picture_provenance[0])
    caption_refs = {caption.cref for caption in picture.captions}
    entries = []
    for index, item in enumerate(document.texts):
        text = clean_text(getattr(item, "text", ""))
        provenance = _provenance_data(item)
        same_page = next((value for value in provenance if value.get("page_no") == page_no), None)
        if not text or same_page is None:
            continue
        item_center = _center(same_page)
        distance = (
            abs(picture_center[0] - item_center[0]) + abs(picture_center[1] - item_center[1])
            if picture_center is not None and item_center is not None
            else float(index)
        )
        entries.append({
            "index": index,
            "ref": item.self_ref,
            "label": str(item.label),
            "text": text,
            "distance": distance,
            "center": item_center,
            "source_locations": [{"doc_item_ref": item.self_ref, **same_page}],
        })
    if not entries:
        return None

    nearest = sorted(entries, key=lambda entry: (entry["distance"], entry["index"]))
    selected = [entry for entry in entries if entry["ref"] in caption_refs]
    flow_entries = []
    if class_name == "flow_chart":
        flow_entries = [entry for entry in entries if _contains(picture_provenance[0], entry["center"])]
        # Parsed flow-chart labels normally lie inside the figure box. If its
        # geometry is absent, retain the existing bounded-nearby fallback.
        selected.extend(flow_entries or nearest[:PICTURE_CONTEXT_MAX_ITEMS])
    elif nearest:
        preceding_headers = [
            entry for entry in entries
            if entry["label"] == "section_header" and entry["index"] <= nearest[0]["index"]
        ]
        if preceding_headers:
            selected.append(preceding_headers[-1])
        selected.extend(nearest)

    unique_entries = []
    seen_refs = set()
    for entry in selected:
        if entry["ref"] in seen_refs:
            continue
        seen_refs.add(entry["ref"])
        unique_entries.append(entry)
        if (class_name != "flow_chart" or not flow_entries) and len(unique_entries) == PICTURE_CONTEXT_MAX_ITEMS:
            break

    parts = []
    used_entries = []
    remaining = max_chars
    for entry in unique_entries:
        part = f"[{entry['label']}] {entry['text']}"
        if len(part) > remaining:
            part = part[:remaining].rstrip()
        if not part:
            break
        parts.append(part)
        used_entries.append(entry)
        remaining -= len(part) + 1
        if remaining <= 0:
            break
    if not parts:
        return None
    # shortcut: proximity uses bounding-box distance, not reading order; upgrade if multi-column PDFs need layout-aware context.
    return {
        "strategy": PICTURE_CONTEXT_STRATEGY,
        "source_item_refs": [entry["ref"] for entry in used_entries],
        "source_locations": [location for entry in used_entries for location in entry["source_locations"]],
        "text": "\n".join(parts),
    }


def media_records(
    document,
    *,
    document_id: str,
    image_dir: Path,
    data_dir: Path,
    include_context: bool = False,
    picture_describer=None,
) -> list[dict]:
    """Return picture records; only caption links are copied into text chunks."""
    records = []
    for index, picture in enumerate(document.pictures):
        picture_hash = sha256_text(picture.self_ref)
        media_id = f"{document_id}:picture:{index}:{picture_hash[:16]}"
        provenance = [item.model_dump(mode="json") for item in picture.prov]
        page_no = provenance[0]["page_no"] if provenance else None
        class_name, confidence = _class_name(picture)
        caption_text = picture.caption_text(document)
        area_fraction = _picture_area_fraction(document, picture)
        context = None
        if include_context:
            # Route from bounded nearby text first. A false flow-chart label
            # must not make an exercise prompt inherit an entire figure box.
            nearby_context = _picture_context(
                document,
                picture,
                max_chars=PICTURE_CONTEXT_MAX_CHARS,
                class_name=None,
            )
            is_flow_chart = p2_picture_kind(
                class_name,
                caption_text,
                (nearby_context or {}).get("text"),
                area_fraction,
            ) == "flow_chart"
            context = _picture_context(
                document,
                picture,
                max_chars=PICTURE_CONTEXT_MAX_CHARS,
                class_name=class_name,
            ) if is_flow_chart else nearby_context
        description = _description_text(picture)
        description_model = getattr(getattr(getattr(picture, "meta", None), "description", None), "created_by", None)
        image_path = None
        image = picture.get_image(document)
        if image is not None:
            path = image_dir / f"{index:04d}-{picture_hash[:16]}.png"
            _save_png(image, path)
            image_path = str(path.relative_to(data_dir))
        p2_result = None
        if picture_describer is not None:
            p2_result = picture_describer.describe(
                image=image,
                class_name=class_name,
                caption_text=caption_text,
                context=context,
                area_fraction=area_fraction,
            )
            description = p2_result["text"]
            description_model = p2_result["model"]
        records.append({
            "schema_version": 1,
            "media_id": media_id,
            "document_id": document_id,
            "picture_ref": picture.self_ref,
            "page_no": page_no,
            "source_locations": provenance,
            "image_path": image_path,
            "image_sha256": sha256_file(path) if image_path else None,
            "caption_refs": [caption.cref for caption in picture.captions],
            "caption_text": caption_text,
            "classification": {"label": class_name, "confidence": confidence},
            "description": {
                "text": description,
                "model": description_model,
            },
            "context": context,
            "p2": {
                **p2_result,
                "context_source_refs": (context or {}).get("source_item_refs", []),
            } if p2_result is not None else None,
            "decision": _decision(
                class_name,
                page_no,
                description,
                instruction_override=bool(p2_result and (
                    p2_result.get("instruction_override") or p2_result.get("action_caption_override")
                )),
            ),
        })
    return records


def media_chunk_records(
    records: list[dict],
    *,
    document_id: str,
    pipeline_run_id: str,
) -> list[dict]:
    """Create embedding-ready text without duplicating image bytes into normal chunks."""
    chunks = []
    for record in records:
        description = record["description"]["text"]
        if not record["decision"]["retrieval_eligible"] or _is_no_retrieval_description(description):
            continue
        context = record.get("context") or {}
        parts = [record["caption_text"], description]
        if context.get("text"):
            parts.append(f"Document context:\n{context['text']}")
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            continue
        content_hash = sha256_text(text)
        chunks.append({
            "schema_version": 1,
            "chunk_id": f"{record['media_id']}:description:{content_hash[:16]}",
            "document_id": document_id,
            "pipeline_run_id": pipeline_run_id,
            "media_id": record["media_id"],
            "text": text,
            "content_sha256": content_hash,
            "source_locations": [*record["source_locations"], *context.get("source_locations", [])],
        })
    return chunks
