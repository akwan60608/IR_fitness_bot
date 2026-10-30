import unittest

from chunking.rag.service import RagService


class _Embeddings:
    def embed_query(self, _question: str) -> list[float]:
        return [1.0, 0.0]


class RagServiceTest(unittest.TestCase):
    def test_answers_from_top_text_records_with_traceable_source_cards(self) -> None:
        service = RagService(
            document_id="sha256:demo-document",
            records=[
                {
                    "record_id": "text-1",
                    "record_type": "text",
                    "text": "深蹲時保持背部中立。",
                    "embedding": [1.0, 0.0],
                    "source_refs": ["#/texts/1"],
                    "page_numbers": [2],
                },
                {
                    "record_id": "media-1",
                    "record_type": "media",
                    "text": "圖片說明",
                    "embedding": [1.0, 0.0],
                    "source_refs": ["#/pictures/1"],
                    "page_numbers": [2],
                },
            ],
            embeddings=_Embeddings(),
            answer=lambda question, context: f"{question}\n{context}",
        )

        result = service.answer("深蹲怎麼做？")

        self.assertEqual(result["status"], "answered")
        self.assertIn("深蹲時保持背部中立。", result["answer"])
        self.assertEqual(result["sources"], [{
            "source_id": "text-1",
            "document_id": "sha256:demo-document",
            "page_numbers": [2],
            "type": "text",
            "excerpt": "深蹲時保持背部中立。",
        }])

    def test_rejects_invalid_questions_without_generating(self) -> None:
        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "text": "深蹲時保持背部中立。",
                "embedding": [1.0, 0.0],
                "source_refs": ["#/texts/1"],
                "page_numbers": [2],
            }],
            embeddings=_Embeddings(),
            answer=lambda _question, _context: self.fail("invalid input must not generate"),
        )

        result = service.answer("   ")

        self.assertEqual(result["status"], "invalid_request")
        self.assertEqual(result["error_code"], "empty_question")

    def test_returns_retrieved_sources_when_generation_fails(self) -> None:
        def fail_answer(_question: str, _context: str) -> str:
            raise RuntimeError("provider unavailable")

        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "text": "深蹲時保持背部中立。",
                "embedding": [1.0, 0.0],
                "source_refs": ["#/texts/1"],
                "page_numbers": [2],
            }],
            embeddings=_Embeddings(),
            answer=fail_answer,
        )

        result = service.answer("深蹲怎麼做？")

        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(result["error_code"], "generation_failed")
        self.assertEqual([source["source_id"] for source in result["sources"]], ["text-1"])


if __name__ == "__main__":
    unittest.main()
