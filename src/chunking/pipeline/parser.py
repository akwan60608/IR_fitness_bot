"""Docling PDF parsing with an explicit accelerator configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
    smolvlm_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureClassificationLabel


PICTURE_CLASSIFIER_MODEL = "docling-project/DocumentFigureClassifier-v2.5"
PICTURE_DESCRIPTION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
PICTURE_DESCRIPTION_AREA_THRESHOLD = 0.01
PicturePromptProfile = Literal["p0", "p1", "p2"]

# Keep all experiment prompts here so every run records an exact, reviewable input.
PICTURE_PROMPTS: dict[PicturePromptProfile, str] = {
    "p0": (
        "Describe the visible actions, body positions, objects, and movements in this image. "
        "Use one or two short English sentences. Do not invent details."
    ),
    "p1": (
        "Describe only visible evidence in English. Photos: action and objects. Charts, tables, and diagrams: "
        "title, axes, units, labels, and visible numbers. Say 'not legible' if unreadable. Do not estimate or infer."
    ),
    # P2 adds per-picture document context after conversion, when it can be
    # recorded with source references. Its VLM prompts are selected per picture
    # by picture_enrichment.py, after Docling finishes classification and OCR.
    "p2": (
        "Post-conversion contextual prompts selected per picture type."
    ),
}


def picture_prompt(profile: PicturePromptProfile) -> str:
    """Return a versioned VLM prompt used by a comparable pipeline run."""
    return PICTURE_PROMPTS[profile]


def build_converter(
    device: AcceleratorDevice | str = AcceleratorDevice.CUDA,
    num_threads: int = 4,
    *,
    picture_mode: Literal["off", "classify", "describe"] = "classify",
    picture_prompt_profile: PicturePromptProfile = "p0",
    ocr_mode: OcrMode | str = OcrMode.DEFAULT,
) -> DocumentConverter:
    """Create a converter; P0/P1 use Docling VLM, while P2 stops after classification."""
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    if picture_mode not in {"off", "classify", "describe"}:
        raise ValueError(f"Unsupported picture mode: {picture_mode}")
    if picture_prompt_profile not in PICTURE_PROMPTS:
        raise ValueError(f"Unsupported picture prompt profile: {picture_prompt_profile}")
    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(device),
        num_threads=num_threads,
    )
    options.do_ocr = True
    # Use ONNX Runtime explicitly: with Docling's onnxruntime extra this exposes
    # CUDAExecutionProvider instead of silently using OcrAutoOptions on CPU.
    options.ocr_options = RapidOcrOptions(
        backend="onnxruntime",
        mode=OcrMode(ocr_mode),
    )
    options.do_table_structure = True
    options.generate_picture_images = picture_mode != "off"
    options.do_picture_classification = picture_mode != "off"
    if picture_mode == "describe" and picture_prompt_profile != "p2":
        description_options = smolvlm_picture_description.model_copy(deep=True)
        description_options.batch_size = 1
        description_options.scale = 2.0
        description_options.picture_area_threshold = PICTURE_DESCRIPTION_AREA_THRESHOLD
        description_options.classification_deny = [
            PictureClassificationLabel.BAR_CODE,
            PictureClassificationLabel.ICON,
            PictureClassificationLabel.LOGO,
            PictureClassificationLabel.PAGE_THUMBNAIL,
            PictureClassificationLabel.QR_CODE,
            PictureClassificationLabel.SIGNATURE,
            PictureClassificationLabel.STAMP,
        ]
        description_options.classification_min_confidence = 0.8
        description_options.prompt = picture_prompt(picture_prompt_profile)
        description_options.generation_config = {"max_new_tokens": 96, "do_sample": False}
        options.do_picture_description = True
        options.picture_description_options = description_options
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(
    pdf_path: str | Path,
    *,
    device: AcceleratorDevice | str = AcceleratorDevice.CUDA,
    num_threads: int = 4,
    picture_mode: Literal["off", "classify", "describe"] = "classify",
    picture_prompt_profile: PicturePromptProfile = "p0",
    ocr_mode: OcrMode | str = OcrMode.DEFAULT,
):
    """Convert one PDF and retain Docling's status and errors for the manifest."""
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError(f"Input must be an existing PDF: {path}")
    return build_converter(
        device,
        num_threads,
        picture_mode=picture_mode,
        picture_prompt_profile=picture_prompt_profile,
        ocr_mode=ocr_mode,
    ).convert(path)


def parse_pdf(
    pdf_path: str | Path,
    *,
    device: AcceleratorDevice | str = AcceleratorDevice.CUDA,
    num_threads: int = 4,
    picture_mode: Literal["off", "classify", "describe"] = "classify",
    picture_prompt_profile: PicturePromptProfile = "p0",
    ocr_mode: OcrMode | str = OcrMode.DEFAULT,
):
    """Backward-compatible helper for callers that only need the document."""
    return convert_pdf(
        pdf_path,
        device=device,
        num_threads=num_threads,
        picture_mode=picture_mode,
        picture_prompt_profile=picture_prompt_profile,
        ocr_mode=ocr_mode,
    ).document
