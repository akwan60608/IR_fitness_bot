# chunking

PDF chunking and retrieval evaluation pipeline built with Docling.

## Setup

```bash
uv sync
```

## Run the pipeline

```bash
uv run chunking --device cpu --num-threads 4
```

Use `--device cuda` when CUDA is available. Input documents and generated artifacts live under `data/` and are intentionally not tracked.

## Evaluate retrieval

```bash
uv run chunking-rag-eval
uv run chunking-ocr-eval --run <run-id> --ground-truth <ground-truth.jsonl>
```

`chunking-rag-eval --answer` requires `GROQ_API_KEY` in the process environment.
