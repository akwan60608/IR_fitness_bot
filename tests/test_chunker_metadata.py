import unittest
from unittest.mock import patch

from chunking.pipeline.chunker import chunk_document
from chunking.pipeline.metadata import table_records


class _Provenance:
    def model_dump(self, mode: str) -> dict:
        self_mode = mode
        return {"page_no": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}, "mode": self_mode}


class _Chunker:
    def __init__(self) -> None:
        self.tokenizer = self

    def chunk(self, dl_doc):
        item = type("Item", (), {"self_ref": "#/texts/3", "label": "text", "prov": [_Provenance()]})()
        meta = type("Meta", (), {"headings": ["第一章"], "doc_items": [item]})()
        return [type("Chunk", (), {"text": "內容", "meta": meta})()]

    def contextualize(self, chunk) -> str:
        return f"第一章\n{chunk.text}"

    def count_tokens(self, *, text: str) -> int:
        return len(text)

    def get_max_tokens(self) -> int:
        return 4


class _Table:
    self_ref = "#/tables/1"
    prov = [_Provenance()]
    data = type("Data", (), {"num_rows": 2, "num_cols": 3})()

    def export_to_markdown(self, document) -> str:
        self.document = document
        return "| 食物 | 蛋白质 |\n| --- | --- |\n| 牛肉 | 25 |"


class _TableDocument:
    tables = [_Table()]


class ChunkerMetadataTest(unittest.TestCase):
    def test_exports_tables_as_a_separate_traceability_artifact(self) -> None:
        records = table_records(_TableDocument(), document_id="sha256:document")

        self.assertEqual(records[0]["table_ref"], "#/tables/1")
        self.assertEqual(records[0]["row_count"], 2)
        self.assertEqual(records[0]["column_count"], 3)
        self.assertIn("牛肉", records[0]["markdown"])

    def test_links_chunk_to_document_run_and_source_location(self) -> None:
        with patch("chunking.pipeline.chunker.HybridChunker", _Chunker):
            chunks = chunk_document(
                object(),
                document_id="sha256:document",
                pipeline_run_id="run-1",
                media_records=[{"media_id": "media-1", "caption_refs": [], "picture_ref": "#/texts/3"}],
            )

        self.assertEqual(chunks[0]["document_id"], "sha256:document")
        self.assertEqual(chunks[0]["pipeline_run_id"], "run-1")
        self.assertEqual(chunks[0]["headings"], ["第一章"])
        self.assertEqual(chunks[0]["source_locations"][0]["page_no"], 2)
        self.assertEqual(chunks[0]["media_ids"], ["media-1"])
        self.assertIn("sha256:document:0:", chunks[0]["chunk_id"])

    def test_fixed_strategy_preserves_source_metadata(self) -> None:
        with patch("chunking.pipeline.chunker.HybridChunker", _Chunker), patch(
            "chunking.pipeline.chunker.HierarchicalChunker", _Chunker
        ):
            chunks = chunk_document(
                object(),
                document_id="sha256:document",
                pipeline_run_id="run-1",
                strategy="fixed",
            )

        self.assertEqual(chunks[0]["text"], "內容")
        self.assertEqual(chunks[0]["raw_text"], "內容")
        self.assertEqual(chunks[0]["headings"], ["第一章"])
        self.assertEqual(chunks[0]["source_locations"][0]["doc_item_ref"], "#/texts/3")


if __name__ == "__main__":
    unittest.main()
