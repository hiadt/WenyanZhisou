from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate a generic dense title index over the public PaSa "
            "paper database. The script never reads RealScholarQuery labels."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Encode titles and create a FAISS index.")
    build.add_argument("--paper_db", required=True)
    build.add_argument("--output_dir", default="indexes/pasa-title-bge-small")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--batch_size", type=int, default=512)
    build.add_argument("--device", default="cuda")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compare lexical, dense and reciprocal-rank-fused title retrieval.",
    )
    evaluate.add_argument("--paper_db", required=True)
    evaluate.add_argument("--index_dir", required=True)
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--output", default="runs/dense_title_retrieval_metrics.json")
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--model", default="")
    evaluate.add_argument("--batch_size", type=int, default=128)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dense_top_k", type=int, default=200)
    evaluate.add_argument("--lexical_top_k", type=int, default=200)
    evaluate.add_argument("--rrf_k", type=int, default=60)
    evaluate.add_argument("--query_prefix", default=DEFAULT_QUERY_PREFIX)

    args = parser.parse_args()
    if args.command == "build":
        build_index(args)
    else:
        evaluate_index(args)


def build_index(args) -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    paper_ids, titles = load_paper_database(args.paper_db)
    if not paper_ids:
        raise ValueError("paper database is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, device=args.device)
    started = time.perf_counter()
    vectors = model.encode(
        titles,
        batch_size=max(1, args.batch_size),
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "titles.faiss"))
    (output_dir / "paper_ids.json").write_text(
        json.dumps(paper_ids, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "model": args.model,
        "paper_count": len(paper_ids),
        "dimension": int(vectors.shape[1]),
        "normalized": True,
        "build_seconds": time.perf_counter() - started,
        "source": str(Path(args.paper_db).resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def evaluate_index(args) -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from wenyan_competition.retrievers import PasaTitleRetriever

    index_dir = Path(args.index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    paper_ids = json.loads((index_dir / "paper_ids.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "titles.faiss"))
    if index.ntotal != len(paper_ids):
        raise ValueError("FAISS rows and paper id metadata have different lengths")

    examples = load_query_examples(args.queries, limit=args.limit)
    model_name = args.model or manifest["model"]
    model = SentenceTransformer(model_name, device=args.device)
    query_texts = [args.query_prefix + example["query"] for example in examples]
    query_vectors = model.encode(
        query_texts,
        batch_size=max(1, args.batch_size),
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    dense_limit = max(100, args.dense_top_k)
    _, dense_rows = index.search(query_vectors, dense_limit)

    lexical = PasaTitleRetriever(
        args.paper_db,
        limit=max(100, args.lexical_top_k),
        min_score=0.0,
    )
    dense_rankings: list[list[str]] = []
    lexical_rankings: list[list[str]] = []
    fused_rankings: list[list[str]] = []
    for example, rows in zip(examples, dense_rows):
        dense_ids = [paper_ids[int(row)] for row in rows if int(row) >= 0]
        lexical_ids = [paper.paper_id for paper in lexical.search(example["query"])]
        dense_rankings.append(dense_ids)
        lexical_rankings.append(lexical_ids)
        fused_rankings.append(
            reciprocal_rank_fusion(
                [lexical_ids, dense_ids],
                weights=[1.0, 1.0],
                rrf_k=max(1, args.rrf_k),
            )
        )

    cutoffs = [20, 50, 100]
    metrics = {
        "query_count": len(examples),
        "index": manifest,
        "settings": {
            "model": model_name,
            "query_prefix": args.query_prefix,
            "dense_top_k": args.dense_top_k,
            "lexical_top_k": args.lexical_top_k,
            "rrf_k": args.rrf_k,
        },
        "lexical": recall_metrics(examples, lexical_rankings, cutoffs),
        "dense": recall_metrics(examples, dense_rankings, cutoffs),
        "rrf_fusion": recall_metrics(examples, fused_rankings, cutoffs),
    }
    metrics["quality_gate"] = quality_gate(metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_paper_database(path: str | Path) -> tuple[list[str], list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    paper_ids: list[str] = []
    titles: list[str] = []
    for paper_id, value in data.items():
        title = str(value.get("title") or value.get("name") or "") if isinstance(value, dict) else str(value or "")
        title = " ".join(title.split())
        if not title:
            continue
        paper_ids.append(normalize_arxiv_id(str(paper_id)))
        titles.append(title)
    return paper_ids, titles


def load_query_examples(path: str | Path, limit: int = 0) -> list[dict]:
    examples: list[dict] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row.get("question") or row.get("query") or "").strip()
            gold = row.get("answer_arxiv_id") or row.get("gold_ids") or []
            gold_ids = sorted({normalize_arxiv_id(str(value)) for value in gold if value})
            if query and gold_ids:
                examples.append({"query": query, "gold_ids": gold_ids})
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("query file contains no examples with arXiv gold ids")
    return examples


def normalize_arxiv_id(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if "v" in value:
        head, suffix = value.rsplit("v", 1)
        if suffix.isdigit():
            value = head
    return value


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float],
    rrf_k: int = 60,
) -> list[str]:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, paper_id in enumerate(ranking, 1):
            normalized = normalize_arxiv_id(paper_id)
            scores[normalized] = scores.get(normalized, 0.0) + weight / (rrf_k + rank)
            best_rank[normalized] = min(best_rank.get(normalized, rank), rank)
    return sorted(scores, key=lambda paper_id: (-scores[paper_id], best_rank[paper_id], paper_id))


def recall_metrics(
    examples: Sequence[dict],
    rankings: Sequence[Sequence[str]],
    cutoffs: Iterable[int],
) -> dict[str, float]:
    cutoffs = list(cutoffs)
    macro = {cutoff: 0.0 for cutoff in cutoffs}
    hits = {cutoff: 0 for cutoff in cutoffs}
    gold_total = 0
    for example, ranking in zip(examples, rankings):
        gold = set(example["gold_ids"])
        gold_total += len(gold)
        for cutoff in cutoffs:
            found = len(gold & set(ranking[:cutoff]))
            macro[cutoff] += found / len(gold)
            hits[cutoff] += found
    count = max(1, len(examples))
    return {
        **{f"macro_recall@{cutoff}": macro[cutoff] / count for cutoff in cutoffs},
        **{f"micro_recall@{cutoff}": hits[cutoff] / max(1, gold_total) for cutoff in cutoffs},
    }


def quality_gate(metrics: dict) -> dict:
    lexical = metrics["lexical"]
    fused = metrics["rrf_fusion"]
    recall_gain = fused["macro_recall@100"] - lexical["macro_recall@100"]
    top20_floor = fused["macro_recall@20"] - lexical["macro_recall@20"]
    accepted = recall_gain >= 0.02 and top20_floor >= -0.005
    return {
        "accepted_for_end_to_end_trial": accepted,
        "rule": "RRF macro Recall@100 gain >= 0.02 and Recall@20 drop <= 0.005",
        "macro_recall@100_gain": recall_gain,
        "macro_recall@20_change": top20_floor,
    }


if __name__ == "__main__":
    main()
