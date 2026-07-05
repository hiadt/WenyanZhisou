from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **_: object):
        return items

from wenyan_competition.agent import AcademicSearchAgent
from wenyan_competition.config import load_config
from wenyan_competition.dataset import load_jsonl, normalize_eval_aliases
from wenyan_competition.metrics import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="runs/eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--fallback_models", action="store_true")
    parser.add_argument("--no_eval_boost", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.no_eval_boost:
        _apply_formal_eval_defaults(config, use_llm=not args.no_llm)
    agent = AcademicSearchAgent(config, use_llm=not args.no_llm, force_fallback_models=args.fallback_models)
    examples = load_jsonl(args.input, limit=args.limit or None)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metric_rows = []
    hit_rows = []
    debug_rows = []
    eval_pool_k = max(args.top_k, 150 if not args.no_eval_boost else 100)

    with pred_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="evaluating"):
            result = agent.search(ex.query, top_k=eval_pool_k, synthesize=False)
            pred_ids = [p.paper_id or p.doi or p.title for p in result.papers]
            pred_aliases = [paper_aliases(p) for p in result.papers]
            hit_report = _hit_report(ex, result.papers, pred_aliases)
            hit_rows.append(hit_report)
            debug_rows.append(_debug_row(ex, result.papers, pred_ids, pred_aliases, hit_report, result.stats))
            row = {
                "precision@20": flexible_precision_at(pred_aliases, ex.gold_items, 20),
                "recall@20": flexible_recall_at(pred_aliases, ex.gold_items, 20),
                "recall@50": flexible_recall_at(pred_aliases, ex.gold_items, 50),
                "recall@100": flexible_recall_at(pred_aliases, ex.gold_items, 100),
                "f1@20": flexible_f1_at(pred_aliases, ex.gold_items, 20),
                "api_calls": float(result.stats.api_calls),
                "llm_calls": float(result.stats.llm_calls),
                "latency_seconds": float(result.stats.latency_seconds),
            }
            metric_rows.append(row)
            f.write(
                json.dumps(
                    {
                        "query": ex.query,
                        "gold_ids": sorted(ex.gold_ids),
                        "gold_items": [sorted(item) for item in ex.gold_items],
                        "pred_ids": pred_ids[: args.top_k],
                        "pred_aliases": [sorted(item) for item in pred_aliases[: args.top_k]],
                        "metrics": row,
                        "result": result.to_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = aggregate(metric_rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "hit_report.json").write_text(json.dumps(hit_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_debug_csv(out_dir / "eval_debug.csv", debug_rows)
    (out_dir / "report.md").write_text(_report(metrics, len(examples)), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _apply_formal_eval_defaults(config, use_llm: bool) -> None:
    """Use the current score-oriented competition setting.

    v18 keeps the PaSa-style multi-source recall path, but it no longer cuts
    recall blindly.  The first pass is compact; weak candidate pools trigger a
    second pass and a small citation expansion.  Ranking uses RRF so BM25,
    embedding, reranker, API and LLM signals each keep a chance to surface gold
    papers.
    """

    config.retrieval.per_query = min(max(config.retrieval.per_query, 40), 45)
    config.retrieval.max_candidates = 420
    config.retrieval.min_candidate_pool_size = max(config.retrieval.min_candidate_pool_size, 180)
    config.retrieval.enable_adaptive_second_pass = True
    config.retrieval.api_timeout_seconds = min(max(config.retrieval.api_timeout_seconds, 8), 12)
    config.retrieval.pasa_title_limit = max(config.retrieval.pasa_title_limit, 220)
    config.retrieval.pasa_title_min_score = min(config.retrieval.pasa_title_min_score, 0.075)
    config.retrieval.max_rounds = 1
    config.retrieval.citation_expand_seeds = 5
    config.retrieval.citation_expand_limit = 40
    config.retrieval.serper_top_k = min(max(config.retrieval.serper_top_k, 10), 12)
    config.retrieval.serper_arxiv_limit = min(max(config.retrieval.serper_arxiv_limit, 18), 24)
    config.retrieval.serper_query_limit = min(config.retrieval.serper_query_limit, 2)
    config.retrieval.serper_query_variants = min(config.retrieval.serper_query_variants, 2)
    config.retrieval.arxiv_query_limit = min(config.retrieval.arxiv_query_limit, 2)
    config.retrieval.arxiv_query_variants = min(config.retrieval.arxiv_query_variants, 2)
    config.retrieval.api_parallelism = min(max(config.retrieval.api_parallelism, 6), 8)
    config.retrieval.enable_api_cache = True
    config.budget.max_api_calls_per_query = 30
    config.ranking.api_weight = 0.08
    config.ranking.bm25_weight = 0.20
    config.ranking.embedding_weight = 0.32
    config.ranking.reranker_weight = 0.32
    config.ranking.authority_weight = 0.03
    config.ranking.recency_weight = 0.02
    config.ranking.diversity_weight = 0.002
    config.ranking.use_rrf = True
    config.ranking.rrf_k = 60
    if use_llm:
        config.budget.max_llm_calls_per_query = 3
        config.ranking.llm_verify_top_n = 40
        config.ranking.llm_verifier_batch_size = max(config.ranking.llm_verifier_batch_size, 20)
        config.ranking.llm_verifier_weight = 0.08
    else:
        config.ranking.llm_verifier_weight = 0.0


def paper_aliases(paper) -> set[str]:
    aliases = set()
    for value in [paper.paper_id, paper.doi, paper.url, paper.title]:
        aliases |= normalize_eval_aliases(value)
    return aliases


def flexible_precision_at(pred_aliases: list[set[str]], gold_items: list[set[str]], k: int) -> float:
    if k <= 0:
        return 0.0
    return _matched_gold_count(pred_aliases, gold_items, k) / k


def flexible_recall_at(pred_aliases: list[set[str]], gold_items: list[set[str]], k: int) -> float:
    if not gold_items:
        return 0.0
    return _matched_gold_count(pred_aliases, gold_items, k) / len(gold_items)


def flexible_f1_at(pred_aliases: list[set[str]], gold_items: list[set[str]], k: int) -> float:
    precision = flexible_precision_at(pred_aliases, gold_items, k)
    recall = flexible_recall_at(pred_aliases, gold_items, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _matched_gold_count(pred_aliases: list[set[str]], gold_items: list[set[str]], k: int) -> int:
    matched_gold = set()
    for aliases in pred_aliases[:k]:
        for idx, gold_aliases in enumerate(gold_items):
            if idx not in matched_gold and _aliases_match(aliases, gold_aliases):
                matched_gold.add(idx)
                break
    return len(matched_gold)


def _hit_report(ex, papers, pred_aliases: list[set[str]]) -> dict:
    hits = []
    for gold_idx, gold_aliases in enumerate(ex.gold_items):
        rank = None
        paper_title = ""
        paper_id = ""
        for pred_idx, aliases in enumerate(pred_aliases, 1):
            if _aliases_match(aliases, gold_aliases):
                rank = pred_idx
                paper = papers[pred_idx - 1]
                paper_title = paper.title
                paper_id = paper.paper_id or paper.doi
                break
        hits.append(
            {
                "gold_index": gold_idx,
                "rank": rank,
                "gold_aliases": sorted(gold_aliases)[:6],
                "paper_id": paper_id,
                "title": paper_title,
            }
        )
    return {
        "query": ex.query,
        "gold_count": len(ex.gold_items),
        "hit_at_20": sum(1 for h in hits if h["rank"] and h["rank"] <= 20),
        "hit_at_50": sum(1 for h in hits if h["rank"] and h["rank"] <= 50),
        "hit_at_100": sum(1 for h in hits if h["rank"] and h["rank"] <= 100),
        "hits": hits,
    }


def _debug_row(ex, papers, pred_ids: list[str], pred_aliases: list[set[str]], hit_report: dict, stats) -> dict:
    ranks = [h["rank"] for h in hit_report["hits"] if h["rank"]]
    missed = [h["gold_aliases"] for h in hit_report["hits"] if not h["rank"]]
    matched_pool = len(ranks)
    gold_count = len(ex.gold_items)
    return {
        "query_id": str(ex.raw.get("qid") or ex.raw.get("id") or ""),
        "query_text": ex.query,
        "gold_ids": json.dumps(sorted(ex.gold_ids), ensure_ascii=False),
        "pred_top20": json.dumps(pred_ids[:20], ensure_ascii=False),
        "pred_top100": json.dumps(pred_ids[:100], ensure_ascii=False),
        "hit@20": hit_report["hit_at_20"],
        "hit@50": hit_report["hit_at_50"],
        "hit@100": hit_report["hit_at_100"],
        "missed_gold_ids": json.dumps(missed, ensure_ascii=False),
        "candidate_pool_size": len(papers),
        "matched_gold_in_pool": matched_pool,
        "oracle_recall@pool": matched_pool / gold_count if gold_count else 0.0,
        "gold_rank_position": json.dumps(ranks, ensure_ascii=False),
        "api_calls": stats.api_calls,
        "llm_calls": stats.llm_calls,
        "latency": stats.latency_seconds,
        "stage_times": json.dumps(getattr(stats, "stage_times", {}), ensure_ascii=False),
    }


def _write_debug_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aliases_match(pred_aliases: set[str], gold_aliases: set[str]) -> bool:
    if pred_aliases & gold_aliases:
        return True
    pred_titles = _title_aliases(pred_aliases)
    gold_titles = _title_aliases(gold_aliases)
    for pred_title in pred_titles:
        for gold_title in gold_titles:
            if _title_fuzzy_match(pred_title, gold_title):
                return True
    return False


def _title_aliases(aliases: set[str]) -> list[str]:
    out = []
    for alias in aliases:
        if alias.startswith("title:"):
            out.append(alias[6:])
    return out


def _title_fuzzy_match(a: str, b: str) -> bool:
    a = _clean_title_alias(a)
    b = _clean_title_alias(b)
    if len(a) < 12 or len(b) < 12:
        return False
    if a == b:
        return True
    shorter, longer = sorted([a, b], key=len)
    if len(shorter) >= 24 and shorter in longer:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.92


def _clean_title_alias(text: str) -> str:
    text = str(text or "").lower()
    text = " ".join(
        token
        for token in text.split()
        if token not in {"arxiv", "preprint", "proceedings", "paper", "article"}
    )
    return " ".join(text.split())


def _report(metrics, n: int) -> str:
    lines = ["# Competition Evaluation Report", "", f"- examples: {n}", ""]
    lines += ["| metric | value |", "|---|---:|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
