import copy
import unittest
from types import SimpleNamespace

from chunking.pipeline.cleaner import clean_document, clean_text


class _Document:
    def __init__(self) -> None:
        self.texts = [SimpleNamespace(text="a  b")]
        description = SimpleNamespace(text="A VLM description")
        self.pictures = [SimpleNamespace(meta=SimpleNamespace(description=description))]

    def model_copy(self, *, deep: bool):
        self.assert_deep = deep
        return copy.deepcopy(self)


class CleanerTest(unittest.TestCase):
    def test_removes_ocr_gaps_inside_cjk_words_and_units_only(self) -> None:
        cleaned = clean_text("离胸部10 厘 米，维持40 秒。\nRAG pipeline 2026 10 28")

        self.assertEqual(cleaned, "离胸部10厘米，维持40秒。\nRAG pipeline 2026 10 28")

    def test_keeps_vlm_text_out_of_regular_chunks_without_mutating_parser_output(self) -> None:
        document = _Document()

        cleaned = clean_document(document)

        self.assertEqual(document.pictures[0].meta.description.text, "A VLM description")
        self.assertEqual(cleaned.pictures[0].meta.description.text, "")
        self.assertEqual(cleaned.texts[0].text, "a b")


if __name__ == "__main__":
    unittest.main()
