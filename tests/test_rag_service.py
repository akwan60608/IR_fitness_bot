import unittest

from chunking.rag.service import RagService


def _cited_answer(text: str, *sources: str) -> dict[str, object]:
    return {"claims": [{"text": text, "sources": list(sources)}]}


class _Embeddings:
    def embed_query(self, _question: str) -> list[float]:
        return [1.0, 0.0]


class _MappedEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.calls.append(question)
        return self._vectors[question]


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
            rewrite=lambda _question: {"retrieval_query": "深蹲怎麼做？", "route": "text"},
            answer=lambda _question, _context, _sources: _cited_answer("深蹲時保持背部中立。", "S1"),
            answerability_threshold=0.0,
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
            "role": "supporting",
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
            rewrite=lambda _question: self.fail("invalid input must not rewrite"),
            answer=lambda _question, _context: self.fail("invalid input must not generate"),
            answerability_threshold=0.0,
        )

        result = service.answer("   ")

        self.assertEqual(result["status"], "invalid_request")
        self.assertEqual(result["error_code"], "empty_question")

    def test_returns_diagnostic_sources_without_generating_when_evidence_is_below_the_gate(self) -> None:
        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "text": "不相干的內容。",
                "embedding": [0.0, 1.0],
                "source_refs": [],
                "page_numbers": [1],
            }],
            embeddings=_Embeddings(),
            rewrite=lambda _question: {"retrieval_query": "問題", "route": "text"},
            answer=lambda _question, _context: self.fail("rejected evidence must not generate"),
            answerability_threshold=0.8,
        )

        result = service.answer("問題")

        self.assertEqual(result["status"], "not_answerable")
        self.assertEqual(result["error_code"], "evidence_gate_rejected")
        self.assertEqual(result["sources"][0]["role"], "diagnostic")

    def test_uses_the_rewritten_query_dense_score_for_the_evidence_gate(self) -> None:
        embeddings = _MappedEmbeddings({"原始問題": [0.0, 1.0], "改寫問題": [1.0, 0.0]})
        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "text": "深蹲時保持背部中立。",
                "embedding": [1.0, 0.0],
                "source_refs": [],
                "page_numbers": [1],
            }],
            embeddings=embeddings,
            rewrite=lambda _question: {"retrieval_query": "改寫問題", "route": "text"},
            answer=lambda _question, _context, _sources: _cited_answer("深蹲時保持背部中立。", "S1"),
            answerability_threshold=0.8,
        )

        result = service.answer("原始問題")

        self.assertEqual(result["status"], "answered")
        self.assertEqual(embeddings.calls, ["原始問題", "改寫問題"])

    def test_returns_retrieved_sources_when_generation_fails(self) -> None:
        def fail_answer(_question: str, _context: str, _sources: tuple[str, ...]) -> dict[str, object]:
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
            rewrite=lambda _question: {"retrieval_query": "深蹲怎麼做？", "route": "text"},
            answer=fail_answer,
            answerability_threshold=0.0,
        )

        result = service.answer("深蹲怎麼做？")

        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(result["error_code"], "generation_failed")
        self.assertEqual([source["source_id"] for source in result["sources"]], ["text-1"])

    def test_rejects_answers_with_missing_or_unknown_source_aliases(self) -> None:
        for answer in (
            {"claims": [{"text": "深蹲時保持背部中立。", "sources": []}]},
            _cited_answer("深蹲時保持背部中立。", "S9"),
            {"claims": [{"text": "深蹲時保持背部中立。", "sources": ["S1"]}, {"text": "", "sources": ["S1"]}]},
        ):
            with self.subTest(answer=answer):
                service = RagService(
                    document_id="sha256:demo-document",
                    records=[{
                        "record_id": "text-1",
                        "record_type": "text",
                        "text": "深蹲時保持背部中立。",
                        "embedding": [1.0, 0.0],
                        "source_refs": [],
                        "page_numbers": [1],
                    }],
                    embeddings=_Embeddings(),
                    rewrite=lambda _question: {"retrieval_query": "深蹲怎麼做？", "route": "text"},
                    answer=lambda _question, _context, _sources: answer,
                    answerability_threshold=0.0,
                )

                result = service.answer("深蹲怎麼做？")

                self.assertEqual(result["status"], "generation_failed")
                self.assertEqual(result["error_code"], "citation_validation_failed")
                self.assertIsNone(result["answer"])
                self.assertEqual(result["sources"][0]["role"], "diagnostic")

    def test_fuses_raw_and_rewritten_top_three_with_rrf(self) -> None:
        embeddings = _MappedEmbeddings({"原始問題": [1.0, 0.0], "改寫問題": [0.0, 1.0]})
        contexts: list[str] = []
        service = RagService(
            document_id="sha256:demo-document",
            records=[
                {
                    "record_id": "raw-a",
                    "record_type": "text",
                    "text": "原始第一名",
                    "embedding": [1.0, 0.0],
                    "source_refs": [],
                    "page_numbers": [1],
                },
                {
                    "record_id": "raw-b",
                    "record_type": "text",
                    "text": "原始第二名",
                    "embedding": [0.9, 0.1],
                    "source_refs": [],
                    "page_numbers": [1],
                },
                {
                    "record_id": "shared",
                    "record_type": "text",
                    "text": "兩個查詢都命中",
                    "embedding": [0.8, 0.8],
                    "source_refs": [],
                    "page_numbers": [1],
                },
                {
                    "record_id": "rewrite-a",
                    "record_type": "text",
                    "text": "改寫第一名",
                    "embedding": [0.0, 1.0],
                    "source_refs": [],
                    "page_numbers": [1],
                },
                {
                    "record_id": "rewrite-b",
                    "record_type": "text",
                    "text": "改寫第二名",
                    "embedding": [0.1, 0.9],
                    "source_refs": [],
                    "page_numbers": [1],
                },
            ],
            embeddings=embeddings,
            rewrite=lambda _question: {"retrieval_query": "改寫問題", "route": "text"},
            answer=lambda _question, context, _sources: contexts.append(context) or _cited_answer("回答", "S1"),
            answerability_threshold=0.0,
        )

        result = service.answer("原始問題")

        self.assertEqual(embeddings.calls, ["原始問題", "改寫問題"])
        self.assertEqual([source["source_id"] for source in result["sources"]], [
            "shared", "raw-a", "rewrite-a",
        ])
        self.assertEqual(result["rewrite"], {
            "status": "rewritten",
            "route": "text",
            "requested_route": "text",
        })
        self.assertEqual(
            contexts,
            [
                "[source_key=S1]\n兩個查詢都命中\n\n"
                "[source_key=S2]\n原始第一名\n\n"
                "[source_key=S3]\n改寫第一名"
            ],
        )

    def test_falls_back_to_the_raw_query_when_rewrite_fails_or_is_invalid(self) -> None:
        for rewrite in (
            lambda _question: (_ for _ in ()).throw(RuntimeError("rewriter unavailable")),
            lambda _question: {"route": "text"},
        ):
            with self.subTest(rewrite=rewrite):
                embeddings = _MappedEmbeddings({"原始問題": [1.0, 0.0]})
                service = RagService(
                    document_id="sha256:demo-document",
                    records=[{
                        "record_id": "text-1",
                        "record_type": "text",
                        "text": "原始查詢結果",
                        "embedding": [1.0, 0.0],
                        "source_refs": [],
                        "page_numbers": [1],
                    }],
                    embeddings=embeddings,
                    rewrite=rewrite,
                    answer=lambda _question, _context, _sources: _cited_answer("回答", "S1"),
                    answerability_threshold=0.0,
                )

                result = service.answer("原始問題")

                self.assertEqual(embeddings.calls, ["原始問題"])
                self.assertEqual(result["rewrite"], {
                    "status": "fallback",
                    "route": "text",
                    "requested_route": "text",
                })
                self.assertEqual(result["status"], "answered")

    def test_reports_a_visual_request_as_text_until_visual_retrieval_exists(self) -> None:
        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "text": "文件文字",
                "embedding": [1.0, 0.0],
                "source_refs": [],
                "page_numbers": [1],
            }],
            embeddings=_Embeddings(),
            rewrite=lambda _question: {"retrieval_query": "圖片問題", "route": "visual"},
            answer=lambda _question, _context, _sources: _cited_answer("回答", "S1"),
            answerability_threshold=0.0,
        )

        result = service.answer("圖片問題")

        self.assertEqual(result["rewrite"], {
            "status": "rewritten",
            "route": "text",
            "requested_route": "visual",
        })

    def test_maps_short_aliases_to_long_record_ids_without_putting_them_in_model_context(self) -> None:
        contexts: list[str] = []
        long_record_id = "sha256:demo-document:chunk:9fbc33a1e2d4"
        service = RagService(
            document_id="sha256:demo-document",
            records=[{
                "record_id": long_record_id,
                "record_type": "text",
                "text": "深蹲時保持背部中立。",
                "embedding": [1.0, 0.0],
                "source_refs": ["#/texts/1"],
                "page_numbers": [2],
            }],
            embeddings=_Embeddings(),
            rewrite=lambda _question: {"retrieval_query": "深蹲怎麼做？", "route": "text"},
            answer=lambda _question, context, _sources: contexts.append(context) or _cited_answer("保持背部中立。", "S1"),
            answerability_threshold=0.0,
        )

        result = service.answer("深蹲怎麼做？")

        self.assertEqual(contexts, ["[source_key=S1]\n深蹲時保持背部中立。"])
        self.assertNotIn(long_record_id, contexts[0])
        self.assertEqual(result["answer"], f"保持背部中立。[source_id={long_record_id}]")
        self.assertEqual(result["sources"][0]["source_id"], long_record_id)


if __name__ == "__main__":
    unittest.main()
