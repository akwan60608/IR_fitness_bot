"""Build a persistent local retrieval index from one pipeline run."""

from __future__ import annotations

import argparse
from pathlib import Path

from chunking.evals.rag import DEFAULT_EMBEDDING_MODEL
from chunking.pipeline.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pipeline-run", required=True, help="Exact successful pipeline_run_id to index.")
    parser.add_argument("--index-dir", type=Path, help="Output directory; defaults to data/index/<pipeline-run>.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    arguments = parser.parse_args()

    index_dir = arguments.index_dir or arguments.data_dir / "index" / arguments.pipeline_run
    manifest = build_index(
        arguments.data_dir,
        arguments.pipeline_run,
        index_dir,
        embedding_model=arguments.embedding_model,
    )
    print(f"Indexed {manifest['document_count']} documents from {manifest['pipeline_run_id']}: {index_dir}")


if __name__ == "__main__":
    main()
