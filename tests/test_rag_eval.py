import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from chunking.evals.rag import (
    _bm25_tokens,
    _is_relevant,
    _metrics,
    evaluate_comparison,
    evaluate_variant,
    load_documents,
    resolve_chunker_runs,
    resolve_runs,
)


class _Embeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def _manifest(profile: str, source: str, completed_at: str, strategy: str = "fixed") -> dict:
    return {
        "pipeline_run_id": f"run-{profile}",
        "status": "success",
        "source_sha256": source,
        "completed_at": completed_at,
        "config": {
            "parser": {"picture": {"mode": "describe", "description_prompt_profile": profile}},
            "chunker": {"strategy": strategy},
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class RagEvaluationTest(unittest.TestCase):
    def test_bm25_uses_chinese_bigrams_and_preserves_latin_numeric_terms(self) -> None:
        self.assertEqual(
            _bm25_tokens("手心向下纏 １５Ｇ Protein"),
            ["手心", "心向", "向下", "下纏", "15g", "protein"],
        )

    def test_document_not_answerable_question_cannot_be_scored_directly(self) -> None:
        question = {
            "pdf_review": {"review_status": "document_not_answerable"},
        }

        with self.assertRaisesRegex(ValueError, "must be excluded"):
            evaluate_variant([], [question], embedding_model="test", top_k=1)

    def test_media_gold_does_not_count_a_regular_chunk(self) -> None:
        question = {"target": "media", "gold": {"picture_refs": ["#/pictures/2"]}}
        text_document = Document(
            page_content="nearby text",
            metadata={"record_type": "text", "picture_ref": "#/pictures/2"},
        )
        media_document = Document(
            page_content="picture description",
            metadata={"record_type": "media", "picture_ref": "#/pictures/2"},
        )

        self.assertFalse(_is_relevant(question, text_document))
        self.assertTrue(_is_relevant(question, media_document))

    def test_metrics_and_vector_search_use_the_gold_source(self) -> None:
        question = {
            "question_id": "q1",
            "category": "text_control",
            "target": "text",
            "question": "short query",
            "reference_answer": "answer",
            "gold": {"doc_item_refs": ["#/texts/2"]},
        }
        documents = [
            Document(
                id="first",
                page_content="a",
                metadata={"record_type": "text", "record_id": "first", "source_refs": ["#/texts/1"], "page_numbers": [1]},
            ),
            Document(
                id="gold",
                page_content="short query answer",
                metadata={"record_type": "text", "record_id": "gold", "source_refs": ["#/texts/2"], "page_numbers": [1]},
            ),
        ]
        with patch("chunking.evals.rag.E5Embeddings", return_value=_Embeddings()):
            results = evaluate_variant(documents, [question], embedding_model="test", top_k=2)

        self.assertEqual(results[0]["first_relevant_rank"], 1)
        self.assertEqual(_metrics(results, 2)["hit_rate_at_1"], 1.0)

    def test_hybrid_search_promotes_a_bm25_match_and_records_both_ranks(self) -> None:
        question = {
            "question_id": "q1",
            "category": "text_control",
            "target": "text",
            "question": "手心向下纏",
            "reference_answer": "手心向下纏。",
            "gold": {
                "doc_item_refs": ["#/texts/2"],
                "hard_negative_doc_item_refs": ["#/texts/1"],
            },
        }
        documents = [
            Document(
                id="distractor",
                page_content="無關文字",
                metadata={"record_type": "text", "record_id": "distractor", "source_refs": ["#/texts/1"], "page_numbers": [1]},
            ),
            Document(
                id="gold",
                page_content="動作說明：手心向下纏繞彈力帶。",
                metadata={"record_type": "text", "record_id": "gold", "source_refs": ["#/texts/2"], "page_numbers": [1]},
            ),
        ]

        results = evaluate_variant(
            documents,
            [question],
            embedding_model="test",
            top_k=2,
            retriever="hybrid",
            embeddings=_Embeddings(),
        )

        retrieved = results[0]["retrieved"][0]
        self.assertEqual(retrieved["record_id"], "gold")
        self.assertEqual(retrieved["retrieval_scores"]["bm25_rank"], 1)
        self.assertIsNotNone(retrieved["retrieval_scores"]["dense_rank"])
        self.assertEqual(results[0]["hard_negative_hits"], [{
            "rank": 2,
            "source_refs": ["#/texts/1"],
            "also_contains_gold": False,
        }])
        self.assertEqual(_metrics(results, 2)["hard_negative_only_at_2_rate"], 1.0)

    def test_auto_selection_requires_a_matching_fixed_p0_p1_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir()
            for profile, completed_at in (("p0", "2026-01-01T00:00:00Z"), ("p1", "2026-01-02T00:00:00Z")):
                (manifest_dir / f"run-{profile}.json").write_text(
                    json.dumps(_manifest(profile, "same-source", completed_at)), encoding="utf-8"
                )

            runs = resolve_runs(data_dir)

        self.assertEqual(runs["p0"]["pipeline_run_id"], "run-p0")
        self.assertEqual(runs["p1"]["pipeline_run_id"], "run-p1")

    def test_pair_rejects_a_non_prompt_configuration_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir()
            p0 = _manifest("p0", "same-source", "2026-01-01T00:00:00Z")
            p1 = _manifest("p1", "same-source", "2026-01-02T00:00:00Z")
            p1["config"]["parser"]["table_structure"] = False
            (manifest_dir / "run-p0.json").write_text(json.dumps(p0), encoding="utf-8")
            (manifest_dir / "run-p1.json").write_text(json.dumps(p1), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "would not be fair"):
                resolve_runs(data_dir)

    def test_explicit_triplet_accepts_p2_context_as_the_experimental_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir()
            for profile, completed_at in (("p0", "2026-01-01T00:00:00Z"), ("p1", "2026-01-02T00:00:00Z"), ("p2", "2026-01-03T00:00:00Z")):
                manifest = _manifest(profile, "same-source", completed_at)
                if profile == "p2":
                    manifest["config"]["parser"]["picture"].update({
                        "retrieval_context_strategy": "same_page_nearby_text_v1",
                        "retrieval_context_max_chars": 1200,
                    })
                (manifest_dir / f"run-{profile}.json").write_text(json.dumps(manifest), encoding="utf-8")

            runs = resolve_runs(data_dir, p0_run="run-p0", p1_run="run-p1", p2_run="run-p2")

        self.assertEqual(tuple(runs), ("p0", "p1", "p2"))

    def test_explicit_p2_run_can_be_evaluated_without_a_p0_p1_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir()
            manifest = _manifest("p2", "same-source", "2026-01-03T00:00:00Z")
            manifest["artifacts"] = {
                "chunks": "chunks/p2.jsonl",
                "media": "media/p2.jsonl",
                "media_chunks": "media_chunks/p2.jsonl",
                "parsed_text": "parsed/p2.jsonl",
            }
            (manifest_dir / "run-p2.json").write_text(json.dumps(manifest), encoding="utf-8")
            _write_jsonl(data_dir / "chunks/p2.jsonl", [{
                "chunk_id": "p2-text", "text": "正確答案", "source_locations": [{"page_no": 1, "doc_item_ref": "#/texts/1"}],
            }])
            _write_jsonl(data_dir / "media/p2.jsonl", [{"media_id": "p2-media", "picture_ref": "#/pictures/1"}])
            _write_jsonl(data_dir / "media_chunks/p2.jsonl", [{
                "chunk_id": "p2-media", "media_id": "p2-media", "text": "圖片描述", "source_locations": [],
            }])
            _write_jsonl(data_dir / "parsed/p2.jsonl", [])
            questions = data_dir / "questions.jsonl"
            _write_jsonl(questions, [{
                "question_id": "q1", "category": "text_control", "target": "text", "question": "問題",
                "reference_answer": "正確答案", "gold": {"doc_item_refs": ["#/texts/1"]},
            }])

            with patch("chunking.evals.rag.E5Embeddings", return_value=_Embeddings()):
                report = evaluate_comparison(data_dir, questions, p2_run="run-p2", embedding_model="test")

        self.assertEqual(tuple(report["variants"]), ("p2",))
        self.assertIn("not a comparison", report["comparison"])
        self.assertEqual(report["config"]["shared_text_chunk_run_id"], "run-p2")

    def test_chunker_pair_accepts_fixed_p0_and_hybrid_p0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir()
            fixed = _manifest("p0", "same-source", "2026-01-01T00:00:00Z")
            hybrid = _manifest("p0", "same-source", "2026-01-02T00:00:00Z", strategy="hybrid")
            fixed["pipeline_run_id"] = "run-fixed"
            hybrid["pipeline_run_id"] = "run-hybrid"
            (manifest_dir / "run-fixed.json").write_text(json.dumps(fixed), encoding="utf-8")
            (manifest_dir / "run-hybrid.json").write_text(json.dumps(hybrid), encoding="utf-8")

            runs = resolve_chunker_runs(data_dir)

        self.assertEqual(runs["fixed"]["pipeline_run_id"], "run-fixed")
        self.assertEqual(runs["hybrid"]["pipeline_run_id"], "run-hybrid")

    def test_p1_reuses_the_p0_text_baseline_but_keeps_its_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            p0_manifest = {"artifacts": {"chunks": "chunks/p0.jsonl"}}
            p1_manifest = {"artifacts": {
                "chunks": "chunks/p1.jsonl",
                "media": "media/p1.jsonl",
                "media_chunks": "media_chunks/p1.jsonl",
            }}
            _write_jsonl(data_dir / "chunks/p0.jsonl", [{
                "chunk_id": "shared-text", "text": "P0 shared text", "source_locations": [],
            }])
            _write_jsonl(data_dir / "chunks/p1.jsonl", [{
                "chunk_id": "p1-text", "text": "P1-only text", "source_locations": [],
            }])
            _write_jsonl(data_dir / "media/p1.jsonl", [{
                "media_id": "media-1", "picture_ref": "#/pictures/1",
            }])
            _write_jsonl(data_dir / "media_chunks/p1.jsonl", [{
                "chunk_id": "p1-media", "media_id": "media-1", "text": "P1 description", "source_locations": [],
            }])

            documents = load_documents(data_dir, p1_manifest, text_manifest=p0_manifest)

        self.assertEqual([document.page_content for document in documents], ["P0 shared text", "P1 description"])

    def test_chunker_comparison_reuses_the_fixed_media_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            fixed_manifest = {"artifacts": {
                "chunks": "chunks/fixed.jsonl",
                "media": "media/fixed.jsonl",
                "media_chunks": "media_chunks/fixed.jsonl",
            }}
            hybrid_manifest = {"artifacts": {
                "chunks": "chunks/hybrid.jsonl",
                "media": "media/hybrid.jsonl",
                "media_chunks": "media_chunks/hybrid.jsonl",
            }}
            _write_jsonl(data_dir / "chunks/fixed.jsonl", [])
            _write_jsonl(data_dir / "chunks/hybrid.jsonl", [{
                "chunk_id": "hybrid-text", "text": "Hybrid text", "source_locations": [],
            }])
            _write_jsonl(data_dir / "media/fixed.jsonl", [{
                "media_id": "fixed-media", "picture_ref": "#/pictures/1",
            }])
            _write_jsonl(data_dir / "media_chunks/fixed.jsonl", [{
                "chunk_id": "fixed-media-chunk", "media_id": "fixed-media", "text": "Fixed description", "source_locations": [],
            }])
            _write_jsonl(data_dir / "media/hybrid.jsonl", [{
                "media_id": "hybrid-media", "picture_ref": "#/pictures/2",
            }])
            _write_jsonl(data_dir / "media_chunks/hybrid.jsonl", [{
                "chunk_id": "hybrid-media-chunk", "media_id": "hybrid-media", "text": "Hybrid description", "source_locations": [],
            }])

            documents = load_documents(data_dir, hybrid_manifest, media_manifest=fixed_manifest)

        self.assertEqual([document.page_content for document in documents], ["Hybrid text", "Fixed description"])


if __name__ == "__main__":
    unittest.main()
