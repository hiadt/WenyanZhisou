from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for smoke tests
    yaml = None


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.2
    max_tokens: int = 900
    timeout: int = 60


@dataclass
class SmallModelConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str = "auto"
    embedding_batch_size: int = 64
    reranker_batch_size: int = 16


@dataclass
class RetrievalConfig:
    use_openalex: bool = True
    use_semantic_scholar: bool = True
    # Resolve short bibliographic queries (title/author/citation-key lookups)
    # through Semantic Scholar's title-match endpoint. Disabled by default so
    # established evaluation profiles keep identical behavior.
    use_semantic_scholar_exact_match: bool = False
    semantic_scholar_exact_query_limit: int = 3
    # Resolve explicit author/year metadata queries through OpenAlex author
    # identities instead of treating names and venues as free-text keywords.
    # Kept opt-in so established PaSa profiles remain identical.
    use_openalex_metadata_constraints: bool = False
    openalex_metadata_author_limit: int = 4
    openalex_metadata_max_year: int = 0
    use_arxiv: bool = True
    # PaSa-style web crawler source.  If enabled and SERPER_API_KEY is set, the
    # retriever searches Google results constrained to arxiv.org, extracts arXiv
    # ids, and then fetches metadata from the official arXiv API.
    use_serper: bool = True
    openalex_mailto: str = ""
    semantic_scholar_api_key: str = ""
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    serper_top_k: int = 10
    serper_arxiv_limit: int = 18
    serper_query_limit: int = 2
    serper_query_variants: int = 2
    arxiv_query_limit: int = 2
    arxiv_query_variants: int = 2
    academic_only: bool = True
    # Optional PaSa paper database: JSON mapping arXiv id -> title.
    # If the file does not exist, the retriever silently skips this source.
    pasa_id2paper_path: str = "data/pasa-dataset/paper_database/id2paper.json"
    pasa_title_limit: int = 80
    pasa_title_min_score: float = 0.10
    # Disabled by default so the accepted v12 path is unchanged until an
    # independently evaluated semantic index is explicitly enabled.
    use_dense_title_index: bool = False
    dense_title_index_path: str = "indexes/pasa-title-bge-small"
    dense_title_limit: int = 200
    dense_title_fusion_limit: int = 120
    dense_title_rrf_k: int = 60
    dense_title_query_prefix: str = "Represent this sentence for searching relevant passages: "
    dense_title_device: str = "auto"
    # Title-only indexes cannot verify author, venue or citation constraints.
    # Keep this opt-in so established evaluation profiles remain unchanged.
    suppress_title_only_for_metadata_queries: bool = False
    # When structured author/year lookup returns verifiable papers, rank that
    # compact set directly instead of mixing it with a large topical pool.
    metadata_direct_shortcut: bool = False
    per_query: int = 20
    max_candidates: int = 120
    citation_expand_seeds: int = 8
    citation_expand_limit: int = 80
    max_rounds: int = 2
    local_corpus_path: str = ""
    local_min_score: float = 0.0
    api_parallelism: int = 6
    enable_api_cache: bool = True
    use_gap_driven_evolution: bool = True
    gap_query_limit: int = 2
    gap_api_reserve: int = 4
    gap_min_coverage: float = 0.25
    early_stop_enabled: bool = True
    early_stop_min_candidates: int = 80
    early_stop_coverage: float = 0.75
    early_stop_budget_fraction: float = 0.85


@dataclass
class RankingConfig:
    api_weight: float = 0.10
    bm25_weight: float = 0.15
    embedding_weight: float = 0.25
    reranker_weight: float = 0.25
    llm_verifier_weight: float = 0.15
    authority_weight: float = 0.06
    recency_weight: float = 0.03
    diversity_weight: float = 0.01
    llm_verify_top_n: int = 30
    llm_verifier_batch_size: int = 20
    constraint_weight: float = 0.04
    constraint_hard_filter_year: bool = True


@dataclass
class BudgetConfig:
    max_llm_calls_per_query: int = 6
    max_api_calls_per_query: int = 30
    max_latency_seconds: int = 180


@dataclass
class AppConfig:
    llm: LLMConfig
    small_models: SmallModelConfig
    retrieval: RetrievalConfig
    ranking: RankingConfig
    budget: BudgetConfig


def load_config(path: str | Path) -> AppConfig:
    raw: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        if yaml is None:
            import json

            raw = json.load(f)
        else:
            raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    return AppConfig(
        llm=LLMConfig(**raw.get("llm", {})),
        small_models=SmallModelConfig(**raw.get("small_models", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        ranking=RankingConfig(**raw.get("ranking", {})),
        budget=BudgetConfig(**raw.get("budget", {})),
    )
