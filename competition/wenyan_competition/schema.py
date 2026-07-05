from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class QueryPlan:
    original_query: str
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    sub_queries: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    full_text: str = ""
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    url: str = ""
    citation_count: int = 0
    source: str = ""
    publication_type: str = ""
    references: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    api_score: float = 0.0
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    reranker_score: float = 0.0
    llm_score: float = 0.0
    authority_score: float = 0.0
    recency_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0
    relevance_label: str = "candidate"
    reason: str = ""

    def text(self) -> str:
        return " ".join(
            [
                self.title or "",
                self.abstract or "",
                (self.full_text or "")[:3000],
                self.venue or "",
            ]
        ).strip()

    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.paper_id:
            return f"id:{self.paper_id}"
        return "title:" + normalize_title(self.title)


@dataclass
class AgentStats:
    llm_calls: int = 0
    api_calls: int = 0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    latency_seconds: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AgentTrace:
    step: int
    role: str
    action: str
    detail: str = ""
    queries: List[str] = field(default_factory=list)
    candidates_before: int = 0
    candidates_after: int = 0
    selected_count: int = 0


@dataclass
class SearchOutput:
    query: str
    plan: QueryPlan
    papers: List[Paper]
    stats: AgentStats
    summary: str = ""
    synthesis: Dict[str, Any] = field(default_factory=dict)
    agent_trace: List[AgentTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "plan": asdict(self.plan),
            "papers": [asdict(p) for p in self.papers],
            "stats": asdict(self.stats),
            "summary": self.summary,
            "synthesis": self.synthesis,
            "agent_trace": [asdict(x) for x in self.agent_trace],
        }


def normalize_title(title: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())).strip()
