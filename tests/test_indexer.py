import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunking.index_cli import main
from chunking.pipeline.indexer import build_index, load_index


class _Embeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 1.0] for index, _text in enumerate(texts)]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _pipeline_artifacts(data_dir: Path, *, status: str = "success") -> str:
    pipeline_run_id = "run-demo"
    manifest = {
        "pipeline_run_id": pipeline_run_id,
        "status": status,
        "document_id": "sha256:demo-document",
        "source_sha256": "a" * 64,
        "artifacts": {
            "chunks": "chunks/demo.jsonl",
            "media": "media/demo.jsonl",
            "media_chunks": "media_chunks/demo.jsonl",
        },
    }
    (data_dir / "manifests").mkdir(parents=True)
    (data_dir / "manifests" / f"{pipeline_run_id}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_jsonl(data_dir / "chunks/demo.jsonl", [{
        "chunk_id": "text-1",
        "text": "深蹲時保持背部中立。",
        "source_locations": [{"page_no": 2, "doc_item_ref": "#/texts/1"}],
    }])
    _write_jsonl(data_dir / "media/demo.jsonl", [{
        "media_id": "media-1",
        "picture_ref": "#/pictures/1",
    }])
    _write_jsonl(data_dir / "media_chunks/demo.jsonl", [{
        "chunk_id": "media-1-description",
        "media_id": "media-1",
        "text": "深蹲姿勢圖片。",
        "source_locations": [{"page_no": 2, "doc_item_ref": "#/pictures/1"}],
    }])
    return pipeline_run_id


class IndexerTest(unittest.TestCase):
    def test_build_and_load_index_preserves_pipeline_and_source_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            index_dir = data_dir / "index" / pipeline_run_id

            manifest = build_index(data_dir, pipeline_run_id, index_dir, embeddings=_Embeddings())
            index = load_index(index_dir)

        self.assertEqual(manifest["pipeline_run_id"], pipeline_run_id)
        self.assertEqual(index.manifest["pipeline_run_id"], pipeline_run_id)
        self.assertEqual([document["record_id"] for document in index.documents], [
            "text-1", "media-1-description",
        ])
        self.assertEqual(index.documents[0]["source_refs"], ["#/texts/1"])
        self.assertEqual(index.documents[0]["page_numbers"], [2])
        self.assertEqual(index.documents[0]["embedding"], [1.0, 1.0])

    def test_build_rejects_a_non_successful_pipeline_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir, status="failed")

            with self.assertRaisesRegex(ValueError, "not successful"):
                build_index(
                    data_dir,
                    pipeline_run_id,
                    data_dir / "index" / pipeline_run_id,
                    embeddings=_Embeddings(),
                )

    def test_build_rejects_a_pipeline_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"

            with self.assertRaisesRegex(ValueError, "invalid pipeline_run_id"):
                build_index(
                    data_dir,
                    "../outside",
                    data_dir / "index" / "outside",
                    embeddings=_Embeddings(),
                )

    def test_build_rejects_a_pipeline_run_without_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            manifest_path = data_dir / "manifests" / f"{pipeline_run_id}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["source_sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_sha256"):
                build_index(
                    data_dir,
                    pipeline_run_id,
                    data_dir / "index" / pipeline_run_id,
                    embeddings=_Embeddings(),
                )

    def test_load_rejects_document_records_that_do_not_match_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            index_dir = data_dir / "index" / pipeline_run_id
            build_index(data_dir, pipeline_run_id, index_dir, embeddings=_Embeddings())
            index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
            documents_path = index_dir / index_manifest["documents_file"]
            records = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()]
            records[0]["index_id"] = "wrong-index"
            _write_jsonl(documents_path, records)

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_index(index_dir)

    def test_load_rejects_non_list_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            index_dir = data_dir / "index" / pipeline_run_id
            index_manifest = build_index(data_dir, pipeline_run_id, index_dir, embeddings=_Embeddings())
            documents_path = index_dir / index_manifest["documents_file"]
            records = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()]
            records[0]["source_refs"] = "#/texts/1"
            _write_jsonl(documents_path, records)

            with self.assertRaisesRegex(ValueError, "source_refs"):
                load_index(index_dir)

    def test_load_rejects_an_invalid_picture_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            index_dir = data_dir / "index" / pipeline_run_id
            index_manifest = build_index(data_dir, pipeline_run_id, index_dir, embeddings=_Embeddings())
            documents_path = index_dir / index_manifest["documents_file"]
            records = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()]
            records[1]["picture_ref"] = {"unexpected": "object"}
            _write_jsonl(documents_path, records)

            with self.assertRaisesRegex(ValueError, "picture_ref"):
                load_index(index_dir)

    def test_load_rejects_a_missing_index_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Missing index artifact"):
                load_index(Path(temporary) / "missing-index")

    def test_cli_builds_an_index_from_a_named_pipeline_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            pipeline_run_id = _pipeline_artifacts(data_dir)
            index_dir = data_dir / "index" / pipeline_run_id

            with (
                patch("chunking.pipeline.indexer.E5Embeddings", return_value=_Embeddings()),
                patch.object(sys, "argv", [
                    "chunking-build-index",
                    "--data-dir", str(data_dir),
                    "--pipeline-run", pipeline_run_id,
                    "--index-dir", str(index_dir),
                    "--embedding-model", "test-model",
                ]),
            ):
                main()

            self.assertEqual(load_index(index_dir).manifest["embedding_model"], "test-model")


if __name__ == "__main__":
    unittest.main()
