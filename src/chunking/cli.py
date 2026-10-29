"""Run PDF parsing, cleanup, and chunking into separate traceability artifacts."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import OcrMode

from chunking.pipeline.parser import (
    PICTURE_CLASSIFIER_MODEL,
    PICTURE_DESCRIPTION_AREA_THRESHOLD,
    PICTURE_DESCRIPTION_MODEL,
    PICTURE_PROMPTS,
    convert_pdf,
    picture_prompt,
)
from chunking.pipeline.media import (
    PICTURE_CONTEXT_MAX_CHARS,
    PICTURE_CONTEXT_STRATEGY,
    media_chunk_records,
    media_records,
    picture_description_record_stats,
    picture_description_stats,
)
from chunking.pipeline.picture_enrichment import P2PictureDescriber, P2_ROUTING_VERSION
from chunking.pipeline.cleaner import clean_document
from chunking.pipeline.chunker import TOKENIZER_NAME, chunk_document, chunker_name


from chunking.pipeline.metadata import (
    document_registry,
    parsed_text_records,
    sha256_file,
    table_records,
    utc_now,
    write_json,
    write_jsonl,
)



def _relative(path: Path, data_dir: Path) -> str:
    return str(path.relative_to(data_dir))


def _effective_device(requested_device: str) -> str:
    if requested_device != AcceleratorDevice.CUDA.value:
        return requested_device
    try:
        import torch
    except ImportError:
        return AcceleratorDevice.CPU.value
    return AcceleratorDevice.CUDA.value if torch.cuda.is_available() else AcceleratorDevice.CPU.value


def _errors(conversion) -> list[dict]:
    return [
        error.model_dump(mode="json") if hasattr(error, "model_dump") else {"message": str(error)}
        for error in conversion.errors
    ]


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return ort.get_available_providers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/half_pdf.pdf"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--device",
        choices=[device.value for device in AcceleratorDevice],
        default=AcceleratorDevice.CUDA.value,
        help="Docling accelerator; CUDA falls back to CPU when unavailable.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--ocr-mode", choices=[mode.value for mode in OcrMode], default=OcrMode.DEFAULT.value)
    parser.add_argument("--picture-mode", choices=("off", "classify", "describe"), default="classify")
    parser.add_argument("--picture-prompt", choices=tuple(PICTURE_PROMPTS), default="p0")
    parser.add_argument("--chunker", choices=("fixed", "hybrid"), default="hybrid")
    arguments = parser.parse_args()

    source_path = arguments.input
    source_hash = sha256_file(source_path)
    document_id = f"sha256:{source_hash}"
    pipeline_run_id = f"run-{uuid4()}"
    effective_device = _effective_device(arguments.device)
    print(f"Requested device: {arguments.device}")
    print(f"Effective device: {effective_device}")
    registry_path = arguments.data_dir / "registry" / f"{source_hash}.json"
    parsed_path = arguments.data_dir / "parsed" / source_hash / f"{pipeline_run_id}.jsonl"
    tables_path = arguments.data_dir / "tables" / source_hash / f"{pipeline_run_id}.jsonl"
    chunks_path = arguments.data_dir / "chunks" / source_hash / f"{pipeline_run_id}.jsonl"
    media_path = arguments.data_dir / "media" / source_hash / f"{pipeline_run_id}.jsonl"
    media_chunks_path = arguments.data_dir / "media_chunks" / source_hash / f"{pipeline_run_id}.jsonl"
    image_dir = media_path.with_suffix("") / "images"
    manifest_path = arguments.data_dir / "manifests" / f"{pipeline_run_id}.json"
    manifest = {
        "schema_version": 1,
        "pipeline_run_id": pipeline_run_id,
        "started_at": utc_now(),
        "document_id": document_id,
        "source_sha256": source_hash,
        "config": {
            "parser": {
                "name": "docling",
                "version": version("docling"),
                "requested_device": arguments.device,
                "device": effective_device,
                "num_threads": arguments.num_threads,
                "ocr": {
                    "enabled": True,
                    "engine": "rapidocr",
                    "backend": "onnxruntime",
                    "mode": arguments.ocr_mode,
                    "onnx_providers": _onnx_providers(),
                },
                "table_structure": True,
                "picture": {
                    "mode": arguments.picture_mode,
                    "generate_images": arguments.picture_mode != "off",
                    "classification_model": PICTURE_CLASSIFIER_MODEL if arguments.picture_mode != "off" else None,
                    "description_model": PICTURE_DESCRIPTION_MODEL if arguments.picture_mode == "describe" else None,
                    "description_prompt_profile": arguments.picture_prompt if arguments.picture_mode == "describe" else None,
                    "description_prompt": picture_prompt(arguments.picture_prompt) if arguments.picture_mode == "describe" and arguments.picture_prompt != "p2" else None,
                    "description_stage": "post_conversion_contextual_v2" if arguments.picture_mode == "describe" and arguments.picture_prompt == "p2" else "docling_builtin",
                    "routing_version": P2_ROUTING_VERSION if arguments.picture_mode == "describe" and arguments.picture_prompt == "p2" else None,
                    "retrieval_context_strategy": PICTURE_CONTEXT_STRATEGY if arguments.picture_mode == "describe" and arguments.picture_prompt == "p2" else None,
                    "retrieval_context_max_chars": PICTURE_CONTEXT_MAX_CHARS if arguments.picture_mode == "describe" and arguments.picture_prompt == "p2" else None,
                },
            },
            "cleaner": {"name": "clean_document", "version": 4},
            "chunker": {"name": chunker_name(arguments.chunker), "strategy": arguments.chunker, "tokenizer": TOKENIZER_NAME},
        },
        "artifacts": {
            "document_registry": _relative(registry_path, arguments.data_dir),
            "parsed_text": _relative(parsed_path, arguments.data_dir),
            "tables": _relative(tables_path, arguments.data_dir),
            "media": _relative(media_path, arguments.data_dir),
            "media_chunks": _relative(media_chunks_path, arguments.data_dir),
            "chunks": _relative(chunks_path, arguments.data_dir),
        },
    }
    try:
        conversion = convert_pdf(
            source_path,
            device=effective_device,
            num_threads=arguments.num_threads,
            picture_mode=arguments.picture_mode,
            picture_prompt_profile=arguments.picture_prompt,
            ocr_mode=arguments.ocr_mode,
        )
        manifest["conversion"] = {"status": conversion.status.value, "errors": _errors(conversion)}
        if conversion.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS} or conversion.document is None:
            raise RuntimeError(f"Docling conversion did not produce a usable document: {conversion.status.value}")
        document = conversion.document
        if not registry_path.exists():
            write_json(registry_path, document_registry(source_path, document_id, source_hash, len(document.pages)))
        write_jsonl(parsed_path, parsed_text_records(document))
        tables = table_records(document, document_id=document_id)
        write_jsonl(tables_path, tables)
        picture_describer = P2PictureDescriber(effective_device) if arguments.picture_mode == "describe" and arguments.picture_prompt == "p2" else None
        records = media_records(
            document,
            document_id=document_id,
            image_dir=image_dir,
            data_dir=arguments.data_dir,
            include_context=arguments.picture_mode == "describe" and arguments.picture_prompt == "p2",
            picture_describer=picture_describer,
        ) if arguments.picture_mode != "off" else []
        write_jsonl(media_path, records)
        media_chunks = media_chunk_records(records, document_id=document_id, pipeline_run_id=pipeline_run_id)
        write_jsonl(media_chunks_path, media_chunks)
        chunks = chunk_document(
            clean_document(document),
            document_id=document_id,
            pipeline_run_id=pipeline_run_id,
            media_records=records,
            strategy=arguments.chunker,
        )
        write_jsonl(chunks_path, chunks)
        manifest.update(
            status=conversion.status.value,
            completed_at=utc_now(),
            counts={
                "parsed_text_items": len(document.texts),
                "tables": len(tables),
                "media": len(records),
                "media_chunks": len(media_chunks),
                "chunks": len(chunks),
                "picture_descriptions": (
                    picture_description_record_stats(records)
                    if arguments.picture_prompt == "p2"
                    else picture_description_stats(
                        document,
                        area_threshold=PICTURE_DESCRIPTION_AREA_THRESHOLD,
                    )
                ) if arguments.picture_mode == "describe" else None,
            },
        )
    except Exception as error:
        manifest.update(status="failed", completed_at=utc_now(), error=str(error))
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(
        f"Generated {len(chunks)} chunks and {len(records)} media records\n"
        f"Registry: {registry_path}\nParsed text: {parsed_path}\nMedia: {media_path}\nManifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
