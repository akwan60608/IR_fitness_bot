# pipeline/cleaner.py

import re


_CJK = "\u3400-\u9fff"


def clean_text(text: str) -> str:
    # 統一換行與特殊空白
    text = text.replace("\r\n", "\n")
    text = text.replace("\u00a0", " ")

    # 移除零寬空白
    text = text.replace("\u200b", "")

    # 清理多餘水平空白，但保留換行
    text = re.sub(r"[ \t]+", " ", text)

    # Scanned CJK PDFs often insert spaces inside a word or before its unit.
    # Keep Latin and numeric-only spacing intact (for example, "2026 10 28").
    text = re.sub(fr"(?<=[{_CJK}])[ \t]+(?=[{_CJK}0-9])", "", text)
    text = re.sub(fr"(?<=\d)[ \t]+(?=[{_CJK}])", "", text)

    # 移除過多空白行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_document(doc):
    """Return a cleaned copy so the parsed document remains traceable."""
    cleaned_doc = doc.model_copy(deep=True)
    for item in cleaned_doc.texts:
        item.text = clean_text(item.text)
    for picture in cleaned_doc.pictures:
        description = getattr(getattr(picture, "meta", None), "description", None)
        if description is not None:
            # VLM text is indexed separately as media_chunks; keep it out of the text-chunker ablation.
            description.text = ""

    return cleaned_doc
