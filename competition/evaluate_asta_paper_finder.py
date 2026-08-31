from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import requests

from evaluate_pasa import _apply_formal_eval_defaults, _remote_llm_requires_key
from wenyan_competition.agent import AcademicSearchAgent
from wenyan_competition.config import load_config
from wenyan_competition.dataset import extract_arxiv_ids
from wenyan_competition.schema import Paper


ASTA_DATASET_REPO = "allenai/asta-bench"
ASTA_DATASET_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"
ASTA_RELEASE = "2025_05"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


@dataclass
class AstaSample:
    query_id: str
    query: str
    scorer_criteria: dict[str, Any]
    raw: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run WenyanZhiSou on AstaBench PaperFindingBench without using "
            "Asta labels inside the search pipeline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download an official AstaBench split.")
    download.add_argument("--split", choices=["validation", "test"], default="validation")
    download.add_argument("--output", default="data/asta-bench/paper_finder_validation.json")
    download.add_argument(
        "--allow_test_download",
        action="store_true",
        help="Required for the test split to prevent accidental test-set tuning.",
    )

    run = subparsers.add_parser("run", help="Run the existing search agent on Asta samples.")
    run.add_argument("--config", default="config.dense.yaml")
    run.add_argument("--input", required=True)
    run.add_argument("--output_dir", default="runs/asta_paper_finder_validation")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--top_k", type=int, default=100)
    run.add_argument("--no_llm", action="store_true")
    run.add_argument("--fallback_models", action="store_true")
    run.add_argument("--no_eval_boost", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        download_split(args.split, Path(args.output), allow_test=args.allow_test_download)
    else:
        run_evaluation(args)


def download_split(split: str, output: Path, *, allow_test: bool = False) -> Path:
    if split == "test" and not allow_test:
        raise ValueError("Test download is disabled during development; use validation instead.")
    token = os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        except (ImportError, OSError):
            token = None
    if not token:
        raise RuntimeError(
            "Log in with `hf auth login`, or set HF_ACCESS_TOKEN/HF_TOKEN, "
            "before downloading AstaBench."
        )
    relative = f"tasks/paper_finder_bench/{split}_{ASTA_RELEASE}.json"
    url = (
        f"https://huggingface.co/datasets/{ASTA_DATASET_REPO}/resolve/"
        f"{ASTA_DATASET_REVISION}/{relative}"
    )
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if response.status_code in {401, 403}:
        raise RuntimeError(
            "AstaBench access was denied. Open the dataset page, accept its access "
            "conditions, and use a read token with access to allenai/asta-bench."
        )
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    print(f"downloaded {split} split to {output} ({len(response.content)} bytes)")
    return output


def run_evaluation(args) -> None:
    config = load_config(args.config)
    if not args.no_llm and _remote_llm_requires_key(config.llm.base_url) and not config.llm.api_key:
        raise RuntimeError("LLM API key is missing; source .env or pass --no_llm.")
    if not args.no_eval_boost:
        _apply_formal_eval_defaults(config, use_llm=not args.no_llm)

    samples = load_asta_samples(args.input, limit=args.limit or None)
    agent = AcademicSearchAgent(
        config,
        use_llm=not args.no_llm,
        force_fallback_models=args.fallback_models,
    )
    resolver = AstaPaperIdResolver(
        api_key=config.retrieval.semantic_scholar_api_key,
        timeout=20,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.jsonl"
    diagnostics: List[dict[str, float]] = []

    with prediction_path.open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, 1):
            started = time.perf_counter()
            result = agent.search(sample.query, top_k=max(args.top_k, 100), synthesize=False)
            selected = result.papers[: args.top_k]
            resolved = resolver.resolve(selected)
            asta_output = build_asta_output(sample.query_id, resolved, limit=args.top_k)
            row_metrics = diagnostic_metrics(
                sample,
                asta_output["output"]["results"],
                retrieved_count=len(selected),
                agent_api_calls=result.stats.api_calls,
                llm_calls=result.stats.llm_calls,
                resolver_api_calls=resolver.last_calls,
                resolver_failures=resolver.last_failures,
                latency_seconds=time.perf_counter() - started,
            )
            diagnostics.append(row_metrics)
            stream.write(
                json.dumps(
                    {
                        "query_id": sample.query_id,
                        "query": sample.query,
                        "query_plan": asdict(result.plan),
                        "agent_trace": [asdict(item) for item in result.agent_trace],
                        "asta_output": asta_output,
                        "diagnostics": row_metrics,
                        "ranked_papers": [
                            {
                                "corpus_id": corpus_id,
                                "title": paper.title,
                                "year": paper.year,
                                "doi": paper.doi,
                                "source": paper.source,
                                "final_score": paper.final_score,
                            }
                            for paper, corpus_id in resolved[: args.top_k]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"[{index}/{len(samples)}] {sample.query_id}: "
                f"resolved={len(asta_output['output']['results'])}/{len(selected)} "
                f"known_hits@30={row_metrics['known_hits@30']:.0f}",
                flush=True,
            )

    summary = aggregate_diagnostics(diagnostics)
    summary["note"] = (
        "Known-good coverage is diagnostic only. Semantic Asta queries require "
        "the official AstaBench LLM relevance scorer for adjusted F1."
    )
    (out_dir / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_asta_samples(path: str | Path, limit: int | None = None) -> List[AstaSample]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        raw_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("dataset"), list):
            raw_rows = payload["dataset"]
        else:
            raise ValueError("Asta input must be a JSON list, a {'dataset': [...]} object, or JSONL.")

    samples: List[AstaSample] = []
    for row in raw_rows:
        input_obj = row.get("input") if isinstance(row, dict) else None
        if not isinstance(input_obj, dict):
            raise ValueError("Asta sample is missing the input object.")
        query_id = str(input_obj.get("query_id") or "").strip()
        query = str(input_obj.get("query") or "").strip()
        if not query_id or not query:
            raise ValueError("Asta sample requires input.query_id and input.query.")
        criteria = row.get("scorer_criteria") or {}
        if not isinstance(criteria, dict):
            raise ValueError("scorer_criteria must be an object.")
        samples.append(AstaSample(query_id, query, criteria, row))
        if limit and len(samples) >= limit:
            break
    return samples


class AstaPaperIdResolver:
    """Resolve heterogeneous paper identifiers to numeric S2 Corpus IDs."""

    def __init__(self, api_key: str = "", timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout
        self.calls = 0
        self.last_calls = 0
        self.failures = 0
        self.last_failures = 0

    def resolve(self, papers: Sequence[Paper]) -> List[tuple[Paper, str]]:
        self.last_calls = 0
        self.last_failures = 0
        corpus_ids: List[str | None] = [direct_corpus_id(paper) for paper in papers]
        lookup_positions: List[int] = []
        lookup_ids: List[str] = []
        for position, (paper, corpus_id) in enumerate(zip(papers, corpus_ids)):
            if corpus_id:
                continue
            lookup_id = semantic_scholar_lookup_id(paper)
            if lookup_id:
                lookup_positions.append(position)
                lookup_ids.append(lookup_id)

        for start in range(0, len(lookup_ids), 100):
            batch_ids = lookup_ids[start : start + 100]
            batch_positions = lookup_positions[start : start + 100]
            rows = self._batch_lookup(batch_ids)
            for position, row in zip(batch_positions, rows):
                if isinstance(row, dict) and row.get("corpusId") is not None:
                    corpus_ids[position] = normalize_corpus_id(row["corpusId"])

        resolved: List[tuple[Paper, str]] = []
        seen = set()
        for paper, corpus_id in zip(papers, corpus_ids):
            if not corpus_id or corpus_id in seen:
                continue
            seen.add(corpus_id)
            resolved.append((paper, corpus_id))
        return resolved

    def _batch_lookup(self, ids: Sequence[str]) -> List[Any]:
        if not ids:
            return []
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        for attempt in range(3):
            self.calls += 1
            self.last_calls += 1
            try:
                response = requests.post(
                    S2_BATCH_URL,
                    params={"fields": "paperId,corpusId,title,externalIds"},
                    headers=headers,
                    json={"ids": list(ids)},
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        retry_after = response.headers.get("Retry-After", "")
                        delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2**attempt
                        time.sleep(min(8.0, max(0.5, delay)))
                        continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) != len(ids):
                    raise ValueError("Semantic Scholar batch response shape is invalid.")
                return payload
            except (requests.RequestException, ValueError):
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
        self.failures += 1
        self.last_failures += 1
        return [None] * len(ids)


def direct_corpus_id(paper: Paper) -> str | None:
    for value in [paper.paper_id, *(paper.source_ids or [])]:
        text = str(value or "").strip()
        match = re.fullmatch(r"(?:corpusid:)?(\d+)", text, flags=re.I)
        if match:
            return normalize_corpus_id(match.group(1))
    return None


def semantic_scholar_lookup_id(paper: Paper) -> str | None:
    if paper.doi:
        return f"DOI:{paper.doi.strip()}"
    for value in [paper.url, paper.paper_id, *(paper.source_ids or [])]:
        arxiv_ids = extract_arxiv_ids(str(value or ""))
        if arxiv_ids:
            return f"ARXIV:{sorted(arxiv_ids)[0]}"
    for value in [paper.paper_id, *(paper.source_ids or [])]:
        text = str(value or "").strip()
        if re.fullmatch(r"[0-9a-f]{40}", text, flags=re.I):
            return text
    return None


def normalize_corpus_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace("corpusid:", "")
    return text if text.isdigit() else ""


def build_asta_output(
    query_id: str,
    resolved: Sequence[tuple[Paper, str]],
    *,
    limit: int,
) -> dict[str, Any]:
    return {
        "output": {
            "query_id": query_id,
            "results": [
                {
                    "paper_id": corpus_id,
                    "markdown_evidence": paper_evidence(paper),
                }
                for paper, corpus_id in resolved[:limit]
            ],
        }
    }


def paper_evidence(paper: Paper) -> str:
    title = " ".join((paper.title or "Untitled").split())
    year = str(paper.year) if paper.year else "unknown"
    parts = [f"Title: {title}", f"Year: {year}"]
    evidence = " ".join((paper.abstract or paper.full_text or "").split())
    if evidence:
        parts.append(f"Abstract: {evidence[:6000]}")
    return "\n\n".join(parts)


def diagnostic_metrics(
    sample: AstaSample,
    results: Sequence[dict[str, Any]],
    *,
    retrieved_count: int,
    agent_api_calls: int,
    llm_calls: int,
    resolver_api_calls: int,
    resolver_failures: int,
    latency_seconds: float,
) -> dict[str, float]:
    predicted = [normalize_corpus_id(row.get("paper_id")) for row in results]
    predicted = [paper_id for paper_id in predicted if paper_id]
    known = extract_known_corpus_ids(sample.scorer_criteria)
    return {
        "retrieved_count": float(retrieved_count),
        "resolved_count": float(len(predicted)),
        "resolution_rate": len(predicted) / retrieved_count if retrieved_count else 0.0,
        "known_count": float(len(known)),
        "known_hits@30": float(len(set(predicted[:30]) & known)),
        "known_coverage@30": len(set(predicted[:30]) & known) / len(known) if known else 0.0,
        "known_coverage@100": len(set(predicted[:100]) & known) / len(known) if known else 0.0,
        "agent_api_calls": float(agent_api_calls),
        "resolver_api_calls": float(resolver_api_calls),
        "resolver_failures": float(resolver_failures),
        "llm_calls": float(llm_calls),
        "latency_seconds": float(latency_seconds),
    }


def extract_known_corpus_ids(criteria: dict[str, Any]) -> set[str]:
    values: Iterable[Any] = criteria.get("corpus_ids") or criteria.get("known_to_be_good") or []
    return {normalized for value in values if (normalized := normalize_corpus_id(value))}


def aggregate_diagnostics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0.0}
    keys = rows[0].keys()
    result = {"samples": float(len(rows))}
    for key in keys:
        result[f"mean_{key}"] = sum(row[key] for row in rows) / len(rows)
    return result


if __name__ == "__main__":
    main()
