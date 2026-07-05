from __future__ import annotations

import argparse
import json
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

    with pred_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="evaluating"):
            result = agent.search(ex.query, top_k=max(args.top_k, 100), synthesize=False)
            pred_ids = [p.paper_id or p.doi or p.title for p in result.papers]
            pred_aliases = [paper_aliases(p) for p in result.papers]
            hit_rows.append(_hit_report(ex, result.papers, pred_aliases))
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
    (out_dir / "report.md").write_text(_report(metrics, len(examples)), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _apply_formal_eval_defaults(config, use_llm: bool) -> None:
    """Use one score-oriented competition setting.

    The official scoring gives F1 70%, efficiency 20%, and structured output
    10%.  This setting keeps PaSa-style multi-source recall and LLM selector
    verification, but avoids expensive second-round/citation expansion by
    default because those added latency without stable gains in RealScholarQuery
    spot checks.
    """

    config.retrieval.per_query = min(config.retrieval.per_query, 18)
    config.retrieval.max_candidates = 220
    config.retrieval.pasa_title_limit = max(config.retrieval.pasa_title_limit, 100)
    config.retrieval.max_rounds = 1
    config.retrieval.citation_expand_seeds = 0
    config.retrieval.citation_expand_limit = 0
    config.retrieval.serper_top_k = min(config.retrieval.serper_top_k, 10)
    config.retrieval.serper_arxiv_limit = min(config.retrieval.serper_arxiv_limit, 16)
    config.retrieval.serper_query_limit = min(config.retrieval.serper_query_limit, 2)
    config.retrieval.serper_query_variants = min(config.retrieval.serper_query_variants, 2)
    config.retrieval.arxiv_query_limit = min(config.retrieval.arxiv_query_limit, 2)
    config.retrieval.arxiv_query_variants = min(config.retrieval.arxiv_query_variants, 2)
    config.retrieval.api_parallelism = max(config.retrieval.api_parallelism, 10)
    config.retrieval.enable_api_cache = True
    config.budget.max_api_calls_per_query = 36
    if use_llm:
        config.budget.max_llm_calls_per_query = 4
        config.ranking.llm_verify_top_n = 60
        config.ranking.llm_verifier_batch_size = max(config.ranking.llm_verifier_batch_size, 20)
        config.ranking.api_weight = max(config.ranking.api_weight, 0.14)
        config.ranking.llm_verifier_weight = max(config.ranking.llm_verifier_weight, 0.22)


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
            if idx not in matched_gold and aliases & gold_aliases:
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
            if aliases & gold_aliases:
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


def _report(metrics, n: int) -> str:
    lines = ["# Competition Evaluation Report", "", f"- examples: {n}", ""]
    lines += ["| metric | value |", "|---|---:|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
