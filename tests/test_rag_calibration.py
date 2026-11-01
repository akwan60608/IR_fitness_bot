import unittest

from chunking.rag.calibration import _source_matched_questions, calibrate_answerability_threshold


class _Embeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_query(self, question: str) -> list[float]:
        return self._vectors[question]


class RagCalibrationTest(unittest.TestCase):
    def test_rejects_calibration_questions_from_another_document(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            _source_matched_questions([{
                "question_id": "other-document",
                "question": "問題",
                "target": "text",
                "source": {"source_sha256": "other-source"},
                "pdf_review": {"review_status": "source_confirmed"},
            }], "expected-source")

    def test_calibrates_the_gate_from_raw_and_rewritten_dense_scores(self) -> None:
        report = calibrate_answerability_threshold(
            records=[{
                "record_id": "text-1",
                "record_type": "text",
                "embedding": [1.0, 0.0],
            }],
            questions=[
                {
                    "question_id": "answerable",
                    "question": "可回答",
                    "target": "text",
                    "pdf_review": {"review_status": "source_confirmed"},
                },
                {
                    "question_id": "unanswerable",
                    "question": "不可回答",
                    "target": "text",
                    "pdf_review": {"review_status": "document_not_answerable"},
                },
            ],
            embeddings=_Embeddings({
                "可回答": [0.0, 1.0],
                "可回答改寫": [1.0, 0.0],
                "不可回答": [0.0, 1.0],
            }),
            rewrite=lambda question: {
                "retrieval_query": "可回答改寫" if question == "可回答" else question,
                "route": "text",
            },
        )

        self.assertEqual(report["answerability_threshold"], 1.0)
        self.assertEqual(report["metrics"], {
            "answerable_total": 1,
            "answerable_accepted": 1,
            "document_not_answerable_total": 1,
            "document_not_answerable_rejected": 1,
        })


if __name__ == "__main__":
    unittest.main()
