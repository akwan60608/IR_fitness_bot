"""Evaluate OCR fidelity and visual retrieval readiness from one pipeline run."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf

from chunking.pipeline.metadata import utc_now, write_json


OCR_TARGET = "ocr"
VISUAL_TARGET = "visual"
TABLE_TARGET = "table"
REQUIRED_ARTIFACTS = ("document_registry", "parsed_text", "media", "media_chunks")


def normalize_text(text: str) -> str:
    """Normalize only Unicode form and whitespace for reproducible OCR scoring."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _artifact_path(data_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    try:
        relative_path = Path(manifest["artifacts"][name])
    except KeyError as error:
        raise ValueError(f"Manifest is missing artifacts.{name}.") from error
    path = (data_dir / relative_path).resolve()
    try:
        path.relative_to(data_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest artifact {name} must remain under {data_dir}.") from error
    if not path.is_file():
        raise ValueError(f"Manifest artifact {name} does not exist: {path}")
    return path


def _optional_artifact_path(data_dir: Path, manifest: dict[str, Any], name: str) -> Path | None:
    if name not in manifest.get("artifacts", {}):
        return None
    return _artifact_path(data_dir, manifest, name)


def _validate_bbox(sample: dict[str, Any]) -> None:
    bbox = sample.get("bbox_normalized")
    if not isinstance(bbox, dict) or set(bbox) != {"l", "t", "r", "b"}:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')} needs bbox_normalized with l/t/r/b.")
    try:
        left, top, right, bottom = (float(bbox[key]) for key in ("l", "t", "r", "b"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')} has a non-numeric bbox.") from error
    if not 0 <= left < right <= 1000 or not 0 <= top < bottom <= 1000:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')} bbox must be a TOPLEFT 0..1000 rectangle.")


def validate_ground_truth(samples: list[dict[str, Any]], source_sha256: str) -> None:
    """Reject a GT that cannot be fairly compared with the selected run."""
    seen_ids: set[str] = set()
    required = {
        "schema_version", "sample_id", "source_sha256", "page_no", "coordinate_origin",
        "bbox_normalized", "target", "region_type", "ground_truth_raw",
        "ground_truth_normalized", "critical_tokens", "annotation_status", "note",
    }
    for sample in samples:
        missing = sorted(required - sample.keys())
        sample_id = sample.get("sample_id", "<unknown>")
        if missing:
            raise ValueError(f"{sample_id} is missing required fields: {', '.join(missing)}.")
        if sample_id in seen_ids or not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample_id must be a unique non-empty string: {sample_id!r}.")
        seen_ids.add(sample_id)
        if sample["source_sha256"] != source_sha256:
            raise ValueError(f"{sample_id} source_sha256 does not match the selected run.")
        if sample["coordinate_origin"] != "TOPLEFT":
            raise ValueError(f"{sample_id} must use TOPLEFT coordinates.")
        if not isinstance(sample["page_no"], int) or sample["page_no"] < 1:
            raise ValueError(f"{sample_id} page_no must be a positive integer.")
        _validate_bbox(sample)
        if sample["target"] not in {OCR_TARGET, VISUAL_TARGET, TABLE_TARGET}:
            raise ValueError(f"{sample_id} target must be 'ocr', 'visual', or 'table'.")
        if not isinstance(sample["ground_truth_raw"], str) or not sample["ground_truth_raw"].strip():
            raise ValueError(f"{sample_id} ground_truth_raw must be non-empty.")
        if sample["ground_truth_normalized"] != normalize_text(sample["ground_truth_raw"]):
            raise ValueError(f"{sample_id} ground_truth_normalized must match the documented normalization.")
        if not isinstance(sample["critical_tokens"], list) or not all(
            isinstance(token, str) and token.strip() for token in sample["critical_tokens"]
        ):
            raise ValueError(f"{sample_id} critical_tokens must be a non-empty string list.")
        if not isinstance(sample["annotation_status"], str) or not sample["annotation_status"].strip():
            raise ValueError(f"{sample_id} annotation_status must be non-empty.")
        if sample["target"] == OCR_TARGET and "expected_facts" in sample:
            raise ValueError(f"{sample_id} OCR samples cannot define expected_facts.")
        if sample["target"] in {VISUAL_TARGET, TABLE_TARGET}:
            facts = sample.get("expected_facts")
            if not isinstance(facts, list) or not facts or not all(isinstance(fact, str) and fact.strip() for fact in facts):
                raise ValueError(f"{sample_id} visual and table samples need non-empty expected_facts.")
        if sample["target"] == TABLE_TARGET and "expected_p2_kind" in sample:
            raise ValueError(f"{sample_id} table samples cannot define expected_p2_kind.")
        if "expected_sequence" in sample:
            if sample["region_type"] != "flowchart":
                raise ValueError(f"{sample_id} expected_sequence is reserved for flowchart samples.")
            sequence = sample["expected_sequence"]
            if not isinstance(sequence, list) or len(sequence) < 2 or not all(
                isinstance(item, str) and item.strip() for item in sequence
            ):
                raise ValueError(f"{sample_id} expected_sequence needs at least two non-empty entries.")
    if not samples:
        raise ValueError("Ground truth must contain at least one sample.")


def _page_dimensions(source_path: Path) -> dict[int, tuple[float, float]]:
    if not source_path.is_file():
        raise ValueError(f"Registered source PDF does not exist: {source_path}")
    with pymupdf.open(source_path) as document:
        return {
            page_number: (float(page.rect.width), float(page.rect.height))
            for page_number, page in enumerate(document, start=1)
        }


def _normalized_center(location: dict[str, Any], dimensions: tuple[float, float]) -> tuple[float, float] | None:
    bbox = location.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        left, top, right, bottom = (float(bbox[key]) for key in ("l", "t", "r", "b"))
    except (KeyError, TypeError, ValueError):
        return None
    width, height = dimensions
    if not width or not height:
        return None
    x = ((left + right) / 2) / width * 1000
    y = (top + bottom) / 2
    origin = bbox.get("coord_origin")
    if origin == "BOTTOMLEFT":
        y = height - y
    elif origin != "TOPLEFT":
        return None
    return x, y / height * 1000


def _within_sample(sample: dict[str, Any], location: dict[str, Any], dimensions: tuple[float, float]) -> bool:
    center = _normalized_center(location, dimensions)
    if center is None:
        return False
    bbox = sample["bbox_normalized"]
    return float(bbox["l"]) <= center[0] <= float(bbox["r"]) and float(bbox["t"]) <= center[1] <= float(bbox["b"])


def _records_in_region(
    records: list[dict[str, Any]],
    sample: dict[str, Any],
    page_dimensions: dict[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    dimensions = page_dimensions.get(sample["page_no"])
    if dimensions is None:
        raise ValueError(f"{sample['sample_id']} points to a page outside the source PDF.")
    selected = []
    for record in records:
        for location in record.get("source_locations", []):
            if location.get("page_no") == sample["page_no"] and _within_sample(sample, location, dimensions):
                selected.append(record)
                break
    return selected


def _levenshtein(expected: str, actual: str) -> int:
    if len(expected) < len(actual):
        expected, actual = actual, expected
    previous = list(range(len(actual) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_character in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[actual_index] + 1,
                previous[actual_index - 1] + (expected_character != actual_character),
            ))
        previous = current
    return previous[-1]


def _token_matches(text: str, tokens: list[str]) -> dict[str, bool]:
    return {token: normalize_text(token) in text for token in tokens}


def _sequence_matches(text: str, sequence: list[str] | None) -> bool | None:
    if not sequence:
        return None
    offset = 0
    for item in sequence:
        position = text.find(normalize_text(item), offset)
        if position < 0:
            return False
        offset = position + len(normalize_text(item))
    return True


def _ocr_result(sample: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    actual_raw = "\n".join(str(record.get("text", "")) for record in records).strip()
    expected = sample["ground_truth_normalized"]
    actual = normalize_text(actual_raw)
    distance = _levenshtein(expected, actual)
    token_matches = _token_matches(actual, sample["critical_tokens"])
    failure_reasons = []
    if not records:
        failure_reasons.append("not_detected")
    if distance:
        failure_reasons.append("text_mismatch")
    if not all(token_matches.values()):
        failure_reasons.append("critical_tokens_missing")
    return {
        "sample_id": sample["sample_id"],
        "target": OCR_TARGET,
        "region_type": sample["region_type"],
        "page_no": sample["page_no"],
        "expected_raw": sample["ground_truth_raw"],
        "expected_normalized": expected,
        "actual_raw": actual_raw,
        "actual_normalized": actual,
        "selected_doc_item_refs": [record.get("doc_item_ref") for record in records],
        "detected": bool(records),
        "edit_distance": distance,
        "expected_characters": len(expected),
        "cer": distance / len(expected),
        "exact_match": actual == expected,
        "critical_tokens": token_matches,
        "failure_reasons": failure_reasons,
    }


def _visual_result(
    sample: dict[str, Any],
    records: list[dict[str, Any]],
    media_chunk_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    matches = []
    representations = []
    descriptions = []
    for record in records:
        media_id = record.get("media_id")
        chunks = media_chunk_by_id.get(media_id, [])
        chunk_text = "\n".join(str(chunk.get("text", "")) for chunk in chunks if chunk.get("text"))
        description = (record.get("description") or {}).get("text")
        # media_chunks already embed the P2 description; counting it twice can
        # make an invalid flow-chart order appear valid across the duplicate.
        if chunk_text or description:
            representations.append(chunk_text or description)
        if description:
            descriptions.append(str(description))
        p2 = record.get("p2") or {}
        matches.append({
            "media_id": media_id,
            "picture_ref": record.get("picture_ref"),
            "classification_label": (record.get("classification") or {}).get("label"),
            "p2_kind": p2.get("kind"),
            "p2_prompt_id": p2.get("prompt_id"),
            "p2_routing_reason": p2.get("routing_reason"),
            "p2_status": p2.get("status"),
            "p2_description": description,
            "media_chunk_text": chunk_text or None,
        })
    representation_actual = normalize_text("\n".join(representations))
    description_actual = normalize_text("\n".join(descriptions))
    fact_matches = _token_matches(representation_actual, sample["expected_facts"])
    description_fact_matches = _token_matches(description_actual, sample["expected_facts"])
    sequence_match = _sequence_matches(representation_actual, sample.get("expected_sequence"))
    expected_p2_kind = sample.get("expected_p2_kind")
    prompt_kind_match = (
        any(match["p2_kind"] == expected_p2_kind for match in matches)
        if expected_p2_kind else None
    )
    failure_reasons = []
    if not matches:
        failure_reasons.append("no_media_record")
    if not all(fact_matches.values()):
        failure_reasons.append("visual_facts_missing")
    if descriptions and not all(description_fact_matches.values()):
        failure_reasons.append("description_facts_missing")
    if sequence_match is False:
        failure_reasons.append("reading_order_mismatch")
    if prompt_kind_match is False:
        failure_reasons.append("prompt_kind_mismatch")
    return {
        "sample_id": sample["sample_id"],
        "target": VISUAL_TARGET,
        "region_type": sample["region_type"],
        "source_kind": sample.get("source_kind"),
        "expected_p2_kind": expected_p2_kind,
        "page_no": sample["page_no"],
        "expected_raw": sample["ground_truth_raw"],
        "expected_facts": sample["expected_facts"],
        "actual_representation": representation_actual,
        "actual_description": description_actual or None,
        "media_matches": matches,
        "detected": bool(matches),
        "fact_matches": fact_matches,
        "fact_coverage": sum(fact_matches.values()) / len(fact_matches),
        "description_fact_matches": description_fact_matches,
        "description_fact_coverage": sum(description_fact_matches.values()) / len(description_fact_matches),
        "contextual_representation_fact_coverage": sum(fact_matches.values()) / len(fact_matches),
        "context_lift": (
            sum(fact_matches.values()) - sum(description_fact_matches.values())
        ) / len(fact_matches),
        "complete_facts": all(fact_matches.values()),
        "reading_order_match": sequence_match,
        "prompt_kind_match": prompt_kind_match,
        "failure_reasons": failure_reasons,
    }


def _table_result(sample: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    representation = normalize_text("\n".join(str(record.get("markdown", "")) for record in records))
    fact_matches = _token_matches(representation, sample["expected_facts"])
    failure_reasons = []
    if not records:
        failure_reasons.append("no_table_record")
    if not all(fact_matches.values()):
        failure_reasons.append("table_facts_missing")
    return {
        "sample_id": sample["sample_id"],
        "target": TABLE_TARGET,
        "region_type": sample["region_type"],
        "source_kind": sample.get("source_kind"),
        "page_no": sample["page_no"],
        "expected_raw": sample["ground_truth_raw"],
        "expected_facts": sample["expected_facts"],
        "actual_representation": representation,
        "table_matches": [{
            "table_id": record.get("table_id"),
            "table_ref": record.get("table_ref"),
            "row_count": record.get("row_count"),
            "column_count": record.get("column_count"),
            "markdown": record.get("markdown"),
        } for record in records],
        "detected": bool(records),
        "fact_matches": fact_matches,
        "fact_coverage": sum(fact_matches.values()) / len(fact_matches),
        "complete_facts": all(fact_matches.values()),
        "failure_reasons": failure_reasons,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics(
    ocr_results: list[dict[str, Any]],
    visual_results: list[dict[str, Any]],
    table_results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_characters = sum(result["expected_characters"] for result in ocr_results)
    edit_distance = sum(result["edit_distance"] for result in ocr_results)
    token_matches = [matched for result in ocr_results for matched in result["critical_tokens"].values()]
    ordered = [result["reading_order_match"] for result in visual_results if result["reading_order_match"] is not None]
    visual_facts = [matched for result in visual_results for matched in result["fact_matches"].values()]
    description_facts = [matched for result in visual_results for matched in result["description_fact_matches"].values()]
    table_facts = [matched for result in table_results for matched in result["fact_matches"].values()]
    return {
        "ocr": {
            "sample_count": len(ocr_results),
            "detection_recall": _mean([float(result["detected"]) for result in ocr_results]),
            "micro_cer": edit_distance / expected_characters if expected_characters else None,
            "macro_cer": _mean([result["cer"] for result in ocr_results]),
            "normalized_exact_match_rate": _mean([float(result["exact_match"]) for result in ocr_results]),
            "critical_token_accuracy": _mean([float(matched) for matched in token_matches]),
            "critical_token_all_correct_rate": _mean([
                float(all(result["critical_tokens"].values())) for result in ocr_results
            ]),
        },
        "visual": {
            "sample_count": len(visual_results),
            "media_detection_recall": _mean([float(result["detected"]) for result in visual_results]),
            "description_fact_coverage": _mean([float(matched) for matched in description_facts]),
            "contextual_representation_fact_coverage": _mean([float(matched) for matched in visual_facts]),
            "context_lift": _mean([result["context_lift"] for result in visual_results]),
            "contextual_complete_facts_rate": _mean([float(result["complete_facts"]) for result in visual_results]),
            "reading_order_pass_rate": _mean([float(value) for value in ordered]),
        },
        "table": {
            "sample_count": len(table_results),
            "table_detection_recall": _mean([float(result["detected"]) for result in table_results]),
            "fact_coverage": _mean([float(matched) for matched in table_facts]),
            "complete_facts_rate": _mean([float(result["complete_facts"]) for result in table_results]),
        },
    }


def evaluate_artifacts(
    *,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    parsed_records: list[dict[str, Any]],
    media_records: list[dict[str, Any]],
    media_chunks: list[dict[str, Any]],
    page_dimensions: dict[int, tuple[float, float]],
    table_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate loaded artifacts; kept pure so tests need no source PDF fixture."""
    if manifest.get("status") != "success":
        raise ValueError("OCR evaluation requires a successful pipeline manifest.")
    source_sha256 = manifest.get("source_sha256")
    if not isinstance(source_sha256, str) or not source_sha256:
        raise ValueError("Manifest needs source_sha256.")
    validate_ground_truth(samples, source_sha256)
    media_chunk_by_id: dict[str, list[dict[str, Any]]] = {}
    for chunk in media_chunks:
        media_id = chunk.get("media_id")
        if isinstance(media_id, str):
            media_chunk_by_id.setdefault(media_id, []).append(chunk)

    ocr_results = []
    visual_results = []
    table_results = []
    table_records = table_records or []
    for sample in samples:
        records = _records_in_region(
            (
                parsed_records if sample["target"] == OCR_TARGET
                else media_records if sample["target"] == VISUAL_TARGET
                else table_records
            ),
            sample,
            page_dimensions,
        )
        if sample["target"] == OCR_TARGET:
            ocr_results.append(_ocr_result(sample, records))
        elif sample["target"] == VISUAL_TARGET:
            visual_results.append(_visual_result(sample, records, media_chunk_by_id))
        else:
            table_results.append(_table_result(sample, records))

    return {
        "schema_version": 1,
        "evaluation_run_id": f"ocr-eval-{manifest['pipeline_run_id']}",
        "evaluated_at": utc_now(),
        "pipeline_run_id": manifest["pipeline_run_id"],
        "source_sha256": source_sha256,
        "parser": manifest.get("config", {}).get("parser", {}),
        "metrics": _metrics(ocr_results, visual_results, table_results),
        "samples": [*ocr_results, *visual_results, *table_results],
    }


def evaluate_run(data_dir: Path, run_id: str, ground_truth_path: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifests" / f"{run_id}.json"
    if not manifest_path.is_file():
        raise ValueError(f"Pipeline manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("pipeline_run_id") != run_id:
        raise ValueError(f"Manifest run ID does not match requested run: {run_id}")
    if manifest.get("status") != "success":
        raise ValueError("OCR evaluation requires a successful pipeline manifest.")
    artifact_paths = {name: _artifact_path(data_dir, manifest, name) for name in REQUIRED_ARTIFACTS}
    tables_path = _optional_artifact_path(data_dir, manifest, "tables")
    registry = _read_json(artifact_paths["document_registry"])
    if registry.get("source_sha256") != manifest.get("source_sha256"):
        raise ValueError("Document registry source_sha256 does not match the manifest.")
    source_path = Path(registry.get("source", {}).get("path", ""))
    report = evaluate_artifacts(
        manifest=manifest,
        samples=_read_jsonl(ground_truth_path),
        parsed_records=_read_jsonl(artifact_paths["parsed_text"]),
        media_records=_read_jsonl(artifact_paths["media"]),
        media_chunks=_read_jsonl(artifact_paths["media_chunks"]),
        page_dimensions=_page_dimensions(source_path),
        table_records=_read_jsonl(tables_path) if tables_path else [],
    )
    report["ground_truth_path"] = str(ground_truth_path)
    report["artifacts"] = {name: str(path.relative_to(data_dir.resolve())) for name, path in artifact_paths.items()}
    if tables_path:
        report["artifacts"]["tables"] = str(tables_path.relative_to(data_dir.resolve()))
    return report


def write_report(report: dict[str, Any], output_dir: Path, ground_truth_path: Path) -> Path:
    safe_ground_truth_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", ground_truth_path.stem)
    path = output_dir / f"{report['evaluation_run_id']}-{safe_ground_truth_name}.json"
    write_json(path, report)
    return path
