"""P2 picture-specific prompting after Docling has supplied OCR and classification."""

from __future__ import annotations

from chunking.pipeline.parser import PICTURE_DESCRIPTION_AREA_THRESHOLD, PICTURE_DESCRIPTION_MODEL


ACTION_CAPTION_MARKERS = (
    "握", "纏", "缠", "推", "拉", "舉", "举", "抱", "伸", "彎", "弯",
    "拋", "抛", "揮", "挥", "砍", "轉", "转", "抬", "蹲", "踏",
)
HARD_NOISE_CLASSES = {"bar_code", "logo", "page_thumbnail", "qr_code", "signature", "stamp"}
CHART_CLASSES = {"bar_chart", "line_chart", "pie_chart", "table"}
P2_ROUTING_VERSION = "context_action_area_v1"


def has_action_caption(caption_text: str | None) -> bool:
    """Keep instructional figures that Docling's document classifier calls an icon."""
    return bool(caption_text and any(marker in caption_text for marker in ACTION_CAPTION_MARKERS))


def _has_action_context(context_text: str | None, area_fraction: float | None) -> bool:
    """Trust nearby instructions only for a figure-sized region, not a tiny icon."""
    return bool(
        context_text
        and area_fraction is not None
        and area_fraction >= PICTURE_DESCRIPTION_AREA_THRESHOLD
        and "[section_header]" in context_text
        and "[list_item]" in context_text
        and any(marker in context_text for marker in ACTION_CAPTION_MARKERS)
    )


def _action_routing_reason(
    caption_text: str | None,
    context_text: str | None,
    area_fraction: float | None,
) -> str | None:
    if has_action_caption(caption_text):
        return "action_caption"
    if _has_action_context(context_text, area_fraction):
        return "context_action_override"
    return None


def p2_picture_kind(
    class_name: str | None,
    caption_text: str | None,
    context_text: str | None = None,
    area_fraction: float | None = None,
) -> str:
    if class_name in HARD_NOISE_CLASSES:
        return "decorative"
    if _action_routing_reason(caption_text, context_text, area_fraction):
        return "exercise_instruction"
    if class_name == "icon":
        return "decorative"
    if class_name == "flow_chart":
        return "flow_chart"
    if class_name in CHART_CLASSES:
        return "chart_or_table"
    if class_name == "photograph":
        return "exercise_instruction"
    return "general_figure"


def p2_prompt(kind: str, caption_text: str | None, context_text: str | None) -> str:
    evidence = f"Caption: {caption_text or 'None'}\nDocument context: {context_text or 'None'}"
    if kind == "exercise_instruction":
        return (
            "Describe this instructional exercise image using only visible evidence and the supplied document labels. "
            "State body position, limb direction, equipment, and visible start or end phase. "
            "Copy useful caption terms exactly. If a detail is not visible, say 'not visible'; never answer yes/no.\n"
            f"{evidence}"
        )
    if kind == "flow_chart":
        return (
            "Describe this flow chart using only visible labels and arrows. List the named nodes, then write only "
            "clearly visible directed links as 'A -> B'. Copy document labels exactly and write 'arrow unclear' when needed.\n"
            f"{evidence}"
        )
    if kind == "chart_or_table":
        return (
            "Describe this chart or table using only visible evidence. State title, axes or columns, units, labels, "
            "and legible values. Copy document labels exactly; do not estimate missing values.\n"
            f"{evidence}"
        )
    return (
        "Describe the visible document figure in one or two factual sentences. Use the caption and document context "
        "only to name visible elements; do not infer unstated details.\n"
        f"{evidence}"
    )


class P2PictureDescriber:
    """Load the existing local SmolVLM once and generate one contextual prompt per picture."""

    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(PICTURE_DESCRIPTION_MODEL)
        self._model = AutoModelForImageTextToText.from_pretrained(
            PICTURE_DESCRIPTION_MODEL,
            dtype=dtype,
        ).to(self._device)
        self._model.eval()

    def describe(
        self,
        *,
        image,
        class_name: str | None,
        caption_text: str | None,
        context: dict | None,
        area_fraction: float | None = None,
    ) -> dict:
        context_text = (context or {}).get("text")
        routing_reason = _action_routing_reason(caption_text, context_text, area_fraction)
        kind = p2_picture_kind(class_name, caption_text, context_text, area_fraction)
        instruction_override = class_name == "icon" and kind == "exercise_instruction"
        result = {
            "kind": kind,
            "prompt_id": f"p2_{kind}_v1",
            "routing_version": P2_ROUTING_VERSION,
            "routing_reason": routing_reason or "classifier",
            "action_caption_override": class_name == "icon" and routing_reason == "action_caption",
            "instruction_override": instruction_override,
            "area_fraction": area_fraction,
            "model": PICTURE_DESCRIPTION_MODEL,
        }
        if kind == "decorative":
            return {**result, "text": None, "status": "skipped_decorative"}
        if image is None:
            return {**result, "text": None, "status": "skipped_no_image"}

        # Persisting the assembled prompt with the media record makes a later
        # description audit independent of whichever code revision is installed.
        prompt = p2_prompt(kind, caption_text, context_text)
        result["prompt"] = prompt
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            chat_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=chat_prompt, images=image, return_tensors="pt").to(self._device)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(**inputs, max_new_tokens=96, do_sample=False)
            text = self._processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0].strip()
        except Exception as error:
            return {**result, "text": None, "status": "failed", "error": str(error)[:240]}
        return {**result, "text": text or None, "status": "generated" if text else "empty"}
