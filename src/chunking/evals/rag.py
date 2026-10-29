"""Evaluate a controlled RAG retrieval comparison from existing artifacts."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from chunking.pipeline.metadata import utc_now, write_json


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_RETRIEVER = "dense"
DEFAULT_BM25_NGRAM = 2
DEFAULT_HYBRID_CANDIDATES = 20
DEFAULT_RRF_K = 60


def _bm25_tokens(text: str, ngram: int = DEFAULT_BM25_NGRAM) -> list[str]:
    """Tokenize CJK runs into character n-grams and retain Latin/numeric terms."""
    if ngram < 1:
        raise ValueError("bm25_ngram must be at least 1.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", normalized):
        if "\u3400" <= part[0] <= "\u9fff":
            tokens.extend(part[index:index + ngram] for index in range(max(1, len(part) - ngram + 1)))
        else:
            tokens.append(part)
    return tokens


def _document_key(document: Document, index: int = 0) -> str:
    return str(document.metadata.get("record_id") or document.id or index)


class _Bm25Index:
    """Small Okapi BM25 index for the evaluator's existing in-memory documents."""

    def __init__(self, documents: list[Document], *, ngram: int) -> None:
        self._documents = documents
        self._ngram = ngram
        self._term_frequencies = [
            Counter(_bm25_tokens(document.page_content, ngram)) for document in documents
        ]
        self._document_frequencies = Counter(
            term for terms in self._term_frequencies for term in terms
        )
        self._average_length = (
            sum(sum(terms.values()) for terms in self._term_frequencies) / len(documents)
        )

    def search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        if limit < 1:
            return []
        query_terms = Counter(_bm25_tokens(query, self._ngram))
        if not query_terms:
            return []
        scores: list[tuple[int, float]] = []
        for index, terms in enumerate(self._term_frequencies):
            length = sum(terms.values())
            score = 0.0
            for term, query_count in query_terms.items():
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1 + (len(self._documents) - self._document_frequencies[term] + 0.5)
                    / (self._document_frequencies[term] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / self._average_length
                )
                score += query_count * idf * frequency * 2.5 / denominator
            if score:
                scores.append((index, score))
        return [
            (self._documents[index], score)
            for index, score in sorted(
                scores,
                key=lambda item: (-item[1], _document_key(self._documents[item[0]], item[0])),
            )[:limit]
        ]


def _hybrid_ranked(
    dense: list[tuple[Document, float]],
    lexical: list[tuple[Document, float]],
    *,
    top_k: int,
) -> list[tuple[Document, dict[str, float | int | None]]]:
    combined: dict[str, dict[str, Any]] = {}
    for source, ranked in (("dense", dense), ("bm25", lexical)):
        for rank, (document, score) in enumerate(ranked, start=1):
            key = _document_key(document, rank)
            entry = combined.setdefault(key, {
                "document": document,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            })
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = score
            entry["rrf_score"] += 1 / (DEFAULT_RRF_K + rank)
    return [
        (entry["document"], {
            "dense_rank": entry["dense_rank"],
            "dense_score": entry["dense_score"],
            "bm25_rank": entry["bm25_rank"],
            "bm25_score": entry["bm25_score"],
            "rrf_score": entry["rrf_score"],
        })
        for _key, entry in sorted(
            combined.items(),
            key=lambda item: (-item[1]["rrf_score"], item[0]),
        )[:top_k]
    ]


class E5Embeddings:
    """Use the model card's query/document prefixes without changing RAG context."""

    def __init__(self, model_name: str) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(f"query: {text}")


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer only from the supplied context. If it is insufficient, say so. "
        "After every factual claim, cite the relevant source ID in square brackets.",
    ),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _is_document_not_answerable(question: dict[str, Any]) -> bool:
    return question.get("pdf_review", {}).get("review_status") == "document_not_answerable"


def _manifest_matches(manifest: dict[str, Any], profile: str, strategy: str = "fixed") -> bool:
    picture = manifest.get("config", {}).get("parser", {}).get("picture", {})
    chunker = manifest.get("config", {}).get("chunker", {})
    return (
        manifest.get("status") == "success"
        and picture.get("mode") == "describe"
        and picture.get("description_prompt_profile") == profile
        and chunker.get("strategy") == strategy
    )


def _comparison_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep the profile-specific retrieval representation out of the config check."""
    config = json.loads(json.dumps(manifest["config"]))
    picture = config["parser"]["picture"]
    picture.pop("description_prompt_profile", None)
    picture.pop("description_prompt", None)
    picture.pop("description_stage", None)
    picture.pop("retrieval_context_strategy", None)
    picture.pop("retrieval_context_max_chars", None)
    return config


def _chunker_comparison_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep the intended chunker change out of the control-variable check."""
    config = json.loads(json.dumps(manifest["config"]))
    config.pop("chunker", None)
    return config


def resolve_runs(
    data_dir: Path,
    *,
    p0_run: str | None = None,
    p1_run: str | None = None,
    p2_run: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Choose a controlled prompt comparison, or one explicit P2 diagnostic run."""
    if p2_run and not p0_run and not p1_run:
        requested_profiles = ("p2",)
    else:
        requested_profiles = ("p0", "p1", "p2") if p2_run else ("p0", "p1")
    supplied_runs = {"p0": p0_run, "p1": p1_run, "p2": p2_run}
    if any(supplied_runs.values()) and not all(supplied_runs[profile] for profile in requested_profiles):
        required = ", ".join(f"--{profile}-run" for profile in requested_profiles)
        raise ValueError(f"Pass {required} together, or omit all of them to auto-select P0/P1.")

    if all(supplied_runs[profile] for profile in requested_profiles):
        manifests = {
            profile: _read_json(data_dir / "manifests" / f"{supplied_runs[profile]}.json")
            for profile in requested_profiles
        }
    else:
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in (data_dir / "manifests").glob("*.json"):
            manifest = _read_json(path)
            for profile in requested_profiles:
                if _manifest_matches(manifest, profile):
                    candidates[profile].append(manifest)
        shared_sources = {
            manifest["source_sha256"]
            for manifest in candidates[requested_profiles[0]]
        }
        for profile in requested_profiles[1:]:
            shared_sources &= {manifest["source_sha256"] for manifest in candidates[profile]}
        if not shared_sources:
            raise ValueError(f"No successful fixed {'/'.join(profile.upper() for profile in requested_profiles)} runs share the same source PDF.")
        newest_source = max(
            shared_sources,
            key=lambda source: max(
                item.get("completed_at", item.get("started_at", ""))
                for profile in requested_profiles
                for item in candidates[profile]
                if item["source_sha256"] == source
            ),
        )
        manifests = {
            profile: max(
                (item for item in candidates[profile] if item["source_sha256"] == newest_source),
                key=lambda item: item.get("completed_at", item.get("started_at", "")),
            )
            for profile in requested_profiles
        }

    for profile, manifest in manifests.items():
        if not _manifest_matches(manifest, profile):
            raise ValueError(f"{manifest.get('pipeline_run_id')} is not a successful fixed/{profile.upper()} run.")
    if len({manifest["source_sha256"] for manifest in manifests.values()}) != 1:
        raise ValueError("P0, P1, and P2 must use the same source_sha256 for a fair comparison.")
    baseline_config = _comparison_config(next(iter(manifests.values())))
    if any(_comparison_config(manifest) != baseline_config for manifest in manifests.values()):
        raise ValueError("Picture strategies differ outside their representation, so the comparison would not be fair.")
    return manifests


def resolve_chunker_runs(
    data_dir: Path,
    *,
    fixed_run: str | None = None,
    hybrid_run: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Choose a fixed/P0 and hybrid/P0 pair with the same parsed input settings."""
    if bool(fixed_run) != bool(hybrid_run):
        raise ValueError("Pass both --fixed-run and --hybrid-run, or neither to auto-select a matching pair.")

    strategies = {"fixed": "fixed", "hybrid": "hybrid"}
    if fixed_run and hybrid_run:
        manifests = {
            name: _read_json(data_dir / "manifests" / f"{run_id}.json")
            for name, run_id in {"fixed": fixed_run, "hybrid": hybrid_run}.items()
        }
    else:
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in (data_dir / "manifests").glob("*.json"):
            manifest = _read_json(path)
            for name, strategy in strategies.items():
                if _manifest_matches(manifest, "p0", strategy):
                    candidates[name].append(manifest)
        shared_sources = {
            manifest["source_sha256"] for manifest in candidates["fixed"]
        } & {
            manifest["source_sha256"] for manifest in candidates["hybrid"]
        }
        if not shared_sources:
            raise ValueError("No successful fixed/P0 and hybrid/P0 runs share the same source PDF.")
        newest_source = max(
            shared_sources,
            key=lambda source: max(
                item.get("completed_at", item.get("started_at", ""))
                for name in strategies
                for item in candidates[name]
                if item["source_sha256"] == source
            ),
        )
        manifests = {
            name: max(
                (item for item in candidates[name] if item["source_sha256"] == newest_source),
                key=lambda item: item.get("completed_at", item.get("started_at", "")),
            )
            for name in strategies
        }

    for name, strategy in strategies.items():
        if not _manifest_matches(manifests[name], "p0", strategy):
            raise ValueError(f"{manifests[name].get('pipeline_run_id')} is not a successful {strategy}/P0 run.")
    if manifests["fixed"]["source_sha256"] != manifests["hybrid"]["source_sha256"]:
        raise ValueError("Fixed and hybrid runs must use the same source_sha256 for a fair comparison.")
    if _chunker_comparison_config(manifests["fixed"]) != _chunker_comparison_config(manifests["hybrid"]):
        raise ValueError("Fixed and hybrid runs differ outside the chunker, so the comparison would not be fair.")
    return manifests


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = _read_jsonl(path)
    ids: set[str] = set()
    for question in questions:
        question_id = question.get("question_id")
        target = question.get("target")
        gold = question.get("gold", {})
        document_not_answerable = _is_document_not_answerable(question)
        if question.get("schema_version", 1) < 2:
            gold.setdefault("doc_item_refs", [])
            gold.setdefault("picture_refs", [])
        if not isinstance(question_id, str) or not question_id or question_id in ids:
            raise ValueError("Each question needs a unique non-empty question_id.")
        if target not in {"text", "media", "mixed"}:
            raise ValueError(f"{question_id} target must be 'text', 'media', or 'mixed'.")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{question_id} needs a non-empty question.")
        if not isinstance(question.get("reference_answer"), str) or not question["reference_answer"].strip():
            raise ValueError(f"{question_id} needs a non-empty reference_answer.")
        for key in ("doc_item_refs", "picture_refs"):
            if not isinstance(gold.get(key), list):
                raise ValueError(f"{question_id} needs gold.{key} as a list.")
        hard_negative_refs = gold.get("hard_negative_doc_item_refs", [])
        if not isinstance(hard_negative_refs, list):
            raise ValueError(f"{question_id} needs gold.hard_negative_doc_item_refs as a list when present.")
        if set(hard_negative_refs) & set(gold["doc_item_refs"]):
            raise ValueError(f"{question_id} cannot use the same source as gold and hard-negative evidence.")
        if not document_not_answerable and target in {"text", "mixed"} and not gold["doc_item_refs"]:
            raise ValueError(f"{question_id} needs gold.doc_item_refs.")
        if not document_not_answerable and target in {"media", "mixed"} and not gold["picture_refs"]:
            raise ValueError(f"{question_id} needs gold.picture_refs.")
        if question.get("schema_version", 1) >= 2:
            for key in ("page_numbers", "evidence_text"):
                if not isinstance(gold.get(key), list):
                    raise ValueError(f"{question_id} needs gold.{key} as a list.")
            if not document_not_answerable and (not gold["page_numbers"] or not gold["evidence_text"]):
                raise ValueError(f"{question_id} needs stable page_numbers and evidence_text.")
            if not isinstance(question.get("answer_facts"), list):
                raise ValueError(f"{question_id} needs answer_facts as a list.")
            if not document_not_answerable and not question["answer_facts"]:
                raise ValueError(f"{question_id} needs non-empty answer_facts for factuality review.")
        if document_not_answerable:
            if any(gold.get(key) for key in ("doc_item_refs", "picture_refs", "page_numbers", "evidence_text")):
                raise ValueError(f"{question_id} cannot include Gold evidence when the document cannot answer it.")
            if question.get("answer_facts"):
                raise ValueError(f"{question_id} cannot include answer_facts when the document cannot answer it.")
            reason = question.get("pdf_review", {}).get("unresolved_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{question_id} needs pdf_review.unresolved_reason when it is not scored.")
        ids.add(question_id)
    return questions


def _artifact_path(data_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    try:
        return data_dir / manifest["artifacts"][name]
    except KeyError as error:
        raise ValueError(f"{manifest['pipeline_run_id']} has no {name} artifact.") from error


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_refs(record: dict[str, Any]) -> list[str]:
    return sorted({
        location["doc_item_ref"]
        for location in record.get("source_locations", [])
        if location.get("doc_item_ref")
    })


def _page_numbers(record: dict[str, Any]) -> list[int]:
    return sorted({
        location["page_no"]
        for location in record.get("source_locations", [])
        if isinstance(location.get("page_no"), int)
    })


def load_documents(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    text_manifest: dict[str, Any] | None = None,
    media_manifest: dict[str, Any] | None = None,
) -> list[Document]:
    """Load existing chunks, optionally holding text or media constant between variants."""
    text_records = _read_jsonl(_artifact_path(data_dir, text_manifest or manifest, "chunks"))
    media_source = media_manifest or manifest
    media_records = _read_jsonl(_artifact_path(data_dir, media_source, "media"))
    media_by_id = {record["media_id"]: record for record in media_records}
    media_chunks = _read_jsonl(_artifact_path(data_dir, media_source, "media_chunks"))
    documents = [
        Document(
            id=record["chunk_id"],
            page_content=record["text"],
            metadata={
                "record_type": "text",
                "record_id": record["chunk_id"],
                "source_refs": _source_refs(record),
                "page_numbers": _page_numbers(record),
            },
        )
        for record in text_records
        if record.get("text", "").strip()
    ]
    for record in media_chunks:
        media = media_by_id.get(record["media_id"])
        if media is None:
            raise ValueError(f"Media chunk {record['chunk_id']} has no matching media record.")
        documents.append(Document(
            id=record["chunk_id"],
            page_content=record["text"],
            metadata={
                "record_type": "media",
                "record_id": record["chunk_id"],
                "media_id": record["media_id"],
                "picture_ref": media["picture_ref"],
                "source_refs": _source_refs(record),
                "page_numbers": _page_numbers(record),
            },
        ))
    if not documents:
        raise ValueError(f"{manifest['pipeline_run_id']} contains no retrieval documents.")
    return documents


def _evidence_matches(question: dict[str, Any], document: Document) -> set[tuple[str, str]]:
    gold = question["gold"]
    matches: set[tuple[str, str]] = set()
    if document.metadata.get("record_type") in {"text", "media"}:
        matches.update(("text", ref) for ref in set(document.metadata.get("source_refs", [])) & set(gold.get("doc_item_refs", [])))
    if document.metadata.get("record_type") == "media":
        picture_ref = document.metadata.get("picture_ref")
        if picture_ref in gold.get("picture_refs", []):
            matches.add(("picture", picture_ref))
    return matches


def _hard_negative_matches(question: dict[str, Any], document: Document) -> set[str]:
    return set(document.metadata.get("source_refs", [])) & set(
        question["gold"].get("hard_negative_doc_item_refs", [])
    )


def _is_relevant(question: dict[str, Any], document: Document) -> bool:
    matches = _evidence_matches(question, document)
    if question["target"] == "text":
        return any(kind == "text" for kind, _ref in matches)
    if question["target"] == "media":
        return any(kind == "picture" for kind, _ref in matches)
    return bool(matches)


def _gold_evidence(question: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        ("text", ref) for ref in question["gold"].get("doc_item_refs", [])
    } | {
        ("picture", ref) for ref in question["gold"].get("picture_refs", [])
    }


def _metrics(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if not results:
        return {"questions": 0, "hit_rate_at_1": 0.0, f"hit_rate_at_{top_k}": 0.0, f"mrr_at_{top_k}": 0.0}
    first_ranks = [result["first_relevant_rank"] for result in results]
    metrics = {
        "questions": len(results),
        "hit_rate_at_1": sum(rank == 1 for rank in first_ranks) / len(results),
        f"hit_rate_at_{top_k}": sum(rank is not None for rank in first_ranks) / len(results),
        f"mrr_at_{top_k}": sum(1 / rank if rank else 0 for rank in first_ranks) / len(results),
        f"mean_evidence_recall_at_{top_k}": sum(result["evidence_recall_at_k"] for result in results) / len(results),
        f"complete_evidence_rate_at_{top_k}": sum(result["complete_evidence_at_k"] for result in results) / len(results),
        "missed_question_ids": [result["question_id"] for result in results if result["first_relevant_rank"] is None],
    }
    confusable = [result for result in results if result["hard_negative_doc_item_refs"]]
    if confusable:
        def hard_negative_rate(limit: int, *, mixed: bool) -> float:
            return sum(
                any(
                    hit["rank"] <= limit and hit["also_contains_gold"] is mixed
                    for hit in result["hard_negative_hits"]
                )
                for result in confusable
            ) / len(confusable)

        metrics.update({
            "hard_negative_questions": len(confusable),
            "hard_negative_only_at_1_rate": hard_negative_rate(1, mixed=False),
            f"hard_negative_only_at_{top_k}_rate": hard_negative_rate(top_k, mixed=False),
            "mixed_evidence_at_1_rate": hard_negative_rate(1, mixed=True),
            f"mixed_evidence_at_{top_k}_rate": hard_negative_rate(top_k, mixed=True),
        })
    return metrics


def _context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[source_id={document.metadata['record_id']}]\n{document.page_content}"
        for document in documents
    )


def _answer(llm: ChatGroq, question: str, documents: list[Document]) -> str:
    return (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
        "question": question,
        "context": _context(documents),
    })


def evaluate_variant(
    documents: list[Document],
    questions: list[dict[str, Any]],
    *,
    embedding_model: str,
    top_k: int,
    retriever: str = DEFAULT_RETRIEVER,
    bm25_ngram: int = DEFAULT_BM25_NGRAM,
    embeddings: E5Embeddings | None = None,
    llm: ChatGroq | None = None,
) -> list[dict[str, Any]]:
    if retriever not in {"dense", "hybrid"}:
        raise ValueError("retriever must be 'dense' or 'hybrid'.")
    if any(_is_document_not_answerable(question) for question in questions):
        raise ValueError("Document-not-answerable questions must be excluded from retrieval metrics.")
    vectorstore = InMemoryVectorStore(embeddings or E5Embeddings(embedding_model))
    vectorstore.add_documents(documents)
    # shortcut: rebuild BM25 per evaluation run; use a persistent index only when the corpus outgrows local evaluation.
    bm25 = _Bm25Index(documents, ngram=bm25_ngram) if retriever == "hybrid" else None
    candidate_k = min(len(documents), max(top_k, DEFAULT_HYBRID_CANDIDATES))
    results = []
    for question in questions:
        # InMemoryVectorStore exposes cosine scores but no normalized relevance-score function.
        dense_ranked = vectorstore.similarity_search_with_score(
            question["question"], k=candidate_k if bm25 else top_k,
        )
        if bm25:
            ranked = _hybrid_ranked(
                dense_ranked,
                bm25.search(question["question"], candidate_k),
                top_k=top_k,
            )
            retrieved = [document for document, _scores in ranked]
        else:
            ranked = [(document, {"dense_score": score}) for document, score in dense_ranked]
            retrieved = [document for document, _scores in ranked]
        first_relevant_rank = next(
            (rank for rank, document in enumerate(retrieved, start=1) if _is_relevant(question, document)),
            None,
        )
        covered_evidence = (
            set().union(*(_evidence_matches(question, document) for document in retrieved))
            if retrieved else set()
        )
        gold_evidence = _gold_evidence(question)
        hard_negative_hits = [
            {
                "rank": rank,
                "source_refs": sorted(hard_negative_matches),
                "also_contains_gold": bool(_evidence_matches(question, document)),
            }
            for rank, document in enumerate(retrieved, start=1)
            if (hard_negative_matches := _hard_negative_matches(question, document))
        ]
        results.append({
            "question_id": question["question_id"],
            "category": question["category"],
            "target": question["target"],
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "gold": question["gold"],
            "answer_facts": question.get("answer_facts", []),
            "hard_negative_doc_item_refs": question["gold"].get("hard_negative_doc_item_refs", []),
            "hard_negative_hits": hard_negative_hits,
            "first_relevant_rank": first_relevant_rank,
            "covered_evidence": sorted({f"{kind}:{ref}" for kind, ref in covered_evidence}),
            "evidence_recall_at_k": len(covered_evidence) / len(gold_evidence),
            "complete_evidence_at_k": covered_evidence == gold_evidence,
            "retrieved": [
                {
                    "rank": rank,
                    "score": scores["rrf_score"] if "rrf_score" in scores else scores["dense_score"],
                    "retrieval_scores": scores,
                    "record_type": document.metadata["record_type"],
                    "record_id": document.metadata["record_id"],
                    "page_numbers": document.metadata["page_numbers"],
                    "source_refs": document.metadata.get("source_refs", []),
                    "picture_ref": document.metadata.get("picture_ref"),
                    "text": document.page_content,
                }
                for rank, (document, scores) in enumerate(ranked, start=1)
            ],
            "answer": _answer(llm, question["question"], retrieved) if llm else None,
        })
    return results


def evaluate_comparison(
    data_dir: Path,
    questions_path: Path,
    *,
    p0_run: str | None = None,
    p1_run: str | None = None,
    p2_run: str | None = None,
    fixed_run: str | None = None,
    hybrid_run: str | None = None,
    comparison_axis: str = "picture_prompt",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3,
    retriever: str = DEFAULT_RETRIEVER,
    bm25_ngram: int = DEFAULT_BM25_NGRAM,
    answer_with_groq: bool = False,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if retriever not in {"dense", "hybrid"}:
        raise ValueError("retriever must be 'dense' or 'hybrid'.")
    if bm25_ngram < 1:
        raise ValueError("bm25_ngram must be at least 1.")
    if comparison_axis == "picture_prompt":
        manifests = resolve_runs(data_dir, p0_run=p0_run, p1_run=p1_run, p2_run=p2_run)
        variant_names = tuple(manifests)
        text_manifest = manifests.get("p0", manifests["p2"])
        document_options = {"text_manifest": text_manifest}
        if len(variant_names) == 1:
            comparison_description = "fixed-size chunking; P2 picture strategy diagnostic (not a comparison)"
        else:
            comparison_description = f"fixed-size chunking; {' versus '.join(name.upper() for name in variant_names)} picture strategies"
        control_run_config = {"shared_text_chunk_run_id": text_manifest["pipeline_run_id"]}
    elif comparison_axis == "chunker":
        manifests = resolve_chunker_runs(data_dir, fixed_run=fixed_run, hybrid_run=hybrid_run)
        variant_names = ("fixed", "hybrid")
        document_options = {"media_manifest": manifests["fixed"]}
        comparison_description = "P0 picture descriptions; fixed-size versus hybrid chunking"
        control_run_config = {"shared_media_chunk_run_id": manifests["fixed"]["pipeline_run_id"]}
    else:
        raise ValueError("comparison_axis must be 'picture_prompt' or 'chunker'.")
    parsed_text_sha256 = {
        variant_name: _file_sha256(_artifact_path(data_dir, manifests[variant_name], "parsed_text"))
        for variant_name in variant_names
    }
    all_questions = load_questions(questions_path)
    questions = [question for question in all_questions if not _is_document_not_answerable(question)]
    not_scored_questions = [
        {
            "question_id": question["question_id"],
            "reason": question["pdf_review"]["unresolved_reason"],
        }
        for question in all_questions
        if _is_document_not_answerable(question)
    ]
    if not questions:
        raise ValueError("The question set has no source-confirmed questions to score.")
    if answer_with_groq and not os.getenv("GROQ_API_KEY"):
        raise ValueError("--answer requires GROQ_API_KEY in the environment.")
    llm = ChatGroq(model=groq_model, temperature=0) if answer_with_groq else None
    embeddings = E5Embeddings(embedding_model)
    variants = {}
    for variant_name in variant_names:
        documents = load_documents(data_dir, manifests[variant_name], **document_options)
        results = evaluate_variant(
            documents,
            questions,
            embedding_model=embedding_model,
            top_k=top_k,
            retriever=retriever,
            bm25_ngram=bm25_ngram,
            embeddings=embeddings,
            llm=llm,
        )
        text_questions = [question for question in questions if question["target"] == "text"]
        media_questions = [question for question in questions if question["target"] == "media"]
        text_results = evaluate_variant(
            [document for document in documents if document.metadata["record_type"] == "text"],
            text_questions,
            embedding_model=embedding_model,
            top_k=top_k,
            retriever=retriever,
            bm25_ngram=bm25_ngram,
            embeddings=embeddings,
        )
        media_results = evaluate_variant(
            [document for document in documents if document.metadata["record_type"] == "media"],
            media_questions,
            embedding_model=embedding_model,
            top_k=top_k,
            retriever=retriever,
            bm25_ngram=bm25_ngram,
            embeddings=embeddings,
        )
        variants[variant_name] = {
            "pipeline_run_id": manifests[variant_name]["pipeline_run_id"],
            "metrics": {
                "overall": _metrics(results, top_k),
                "text_control": _metrics(text_results, top_k),
                "media": _metrics(media_results, top_k),
                "by_category": {
                    category: _metrics(
                        [result for result in results if result["category"] == category],
                        top_k,
                    )
                    for category in sorted({question["category"] for question in questions})
                },
            },
            "results": results,
            "diagnostic_results": {
                "text_control": text_results,
                "media": media_results,
            },
        }
    return {
        "schema_version": 2,
        "evaluation_run_id": f"rag-eval-{uuid4()}",
        "created_at": utc_now(),
        "comparison": comparison_description,
        "config": {
            "comparison_axis": comparison_axis,
            "embedding_model": embedding_model,
            "top_k": top_k,
            "retrieval": {
                "strategy": retriever,
                "bm25_analyzer": "cjk_character_ngram_plus_alphanumeric" if retriever == "hybrid" else None,
                "bm25_ngram": bm25_ngram if retriever == "hybrid" else None,
                "hybrid_candidates": DEFAULT_HYBRID_CANDIDATES if retriever == "hybrid" else None,
                "rrf_k": DEFAULT_RRF_K if retriever == "hybrid" else None,
            },
            "groq_model": groq_model if answer_with_groq else None,
            "answers_generated": answer_with_groq,
            "parsed_text_sha256": parsed_text_sha256,
            "parser_output_equivalent": len(set(parsed_text_sha256.values())) == 1,
            **control_run_config,
        },
        "questions_path": str(questions_path),
        "question_set": {
            "total_questions": len(all_questions),
            "scored_questions": len(questions),
            "not_scored_document_not_answerable": not_scored_questions,
        },
        "source_sha256": manifests[variant_names[0]]["source_sha256"],
        "variants": variants,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / f"{report['evaluation_run_id']}.json"
    write_json(output_path, report)
    return output_path
