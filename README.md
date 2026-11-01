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

## Run the local RAG service

The service loads exactly one index and defaults its answerability threshold to
`0.6`. Calibrate a source-matched threshold before a real demo, retain the
calibration record, and do not reuse it across different indexes.

```bash
uv run chunking-rag-calibrate \
  --index-dir data/index/<pipeline-run-id> \
  --questions data/evals/questions/pic_pdf_strength_training_v1.jsonl \
  | tee data/evals/reports/gate-calibration-<pipeline-run-id>.json
```

The command uses the same raw-plus-rewritten dense score as the service and
records the question-file hash with the selected `answerability_threshold`.
Recalibrate when that hash or the selected index changes, then override the
default in the service configuration:

```bash
export INDEX_DIR=data/index/<pipeline-run-id>
export ANSWERABILITY_THRESHOLD=<calibrated-dense-cosine-score>
uv run chunking-rag-api
```

The question file must contain source-confirmed and `document_not_answerable`
text questions for that index's source document. The gate uses the highest
dense cosine score from the raw or rewritten query, never the RRF rank.
