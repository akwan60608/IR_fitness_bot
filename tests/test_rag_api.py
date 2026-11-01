import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from chunking.rag.api import _answer_schema, create_app
from chunking.rag.service import RagService
from chunking.rag.streamlit_app import _answer


class _SseResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


class _Embeddings:
    def embed_query(self, _question: str) -> list[float]:
        return [1.0, 0.0]


def _cited_answer(text: str, *sources: str) -> dict[str, object]:
    return {"claims": [{"text": text, "sources": list(sources)}]}


class RagApiTest(unittest.TestCase):
    def test_answer_schema_limits_citations_to_the_request_aliases(self) -> None:
        schema = _answer_schema(("S1", "S2"))
        source_items = schema["parameters"]["properties"]["claims"]["items"]["properties"]["sources"]["items"]

        self.assertEqual(source_items["enum"], ["S1", "S2"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertFalse(schema["parameters"]["properties"]["claims"]["items"]["additionalProperties"])

    def test_streamlit_client_rejects_malformed_or_missing_result_events(self) -> None:
        with patch("chunking.rag.streamlit_app.urlopen", return_value=_SseResponse([b"data: []\n"])):
            with self.assertRaisesRegex(ValueError, "invalid"):
                _answer("http://example.test", "問題")
        with patch(
            "chunking.rag.streamlit_app.urlopen",
            return_value=_SseResponse([b'data: {"status": "answered", "sources": []}\n']),
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                _answer("http://example.test", "問題")
        with patch("chunking.rag.streamlit_app.urlopen", return_value=_SseResponse([])):
            with self.assertRaisesRegex(ValueError, "result"):
                _answer("http://example.test", "問題")

    def test_startup_requires_an_explicit_index_directory(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            from chunking.rag.api import create_app_from_environment

            with self.assertRaisesRegex(RuntimeError, "INDEX_DIR"):
                create_app_from_environment()

    def test_answerability_threshold_defaults_to_point_six_but_rejects_invalid_overrides(self) -> None:
        for value, expected in ((None, 0.6), ("not-a-score", None), ("1.1", None)):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {} if value is None else {"ANSWERABILITY_THRESHOLD": value},
                clear=True,
            ):
                from chunking.rag.api import _answerability_threshold_from_environment

                if expected is None:
                    with self.assertRaisesRegex(RuntimeError, "ANSWERABILITY_THRESHOLD"):
                        _answerability_threshold_from_environment()
                else:
                    self.assertEqual(_answerability_threshold_from_environment(), expected)

    def test_answer_endpoint_returns_sse_result_and_done_events(self) -> None:
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
            answer=lambda _question, _context, _sources: _cited_answer("保持背部中立。", "S1"),
            answerability_threshold=0.0,
        )

        response = TestClient(create_app(service)).post("/answer", json={"question": "深蹲怎麼做？"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: result", response.text)
        self.assertIn('"status": "answered"', response.text)
        self.assertIn(
            '"rewrite": {"status": "rewritten", "route": "text", "requested_route": "text"}',
            response.text,
        )
        self.assertTrue(response.text.endswith("event: done\ndata: {}\n\n"))

    def test_answer_endpoint_requires_a_question_field(self) -> None:
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
            answer=lambda _question, _context, _sources: _cited_answer("保持背部中立。", "S1"),
            answerability_threshold=0.0,
        )

        response = TestClient(create_app(service)).post("/answer", json={})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(json.loads(response.text)["detail"][0]["loc"], ["body", "question"])


if __name__ == "__main__":
    unittest.main()
