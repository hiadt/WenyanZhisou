from __future__ import annotations

import re
import time
from collections import Counter
from typing import Dict, List, Optional

from .config import AppConfig
from .llm import LLMClient, LLMPlanner, LLMQueryEvolver, LLMVerifier, ResultSynthesizer
from .ranker import CompetitionRanker
from .retrievers import AcademicRetriever, deduplicate
from .schema import AgentStats, AgentTrace, Paper, QueryPlan, SearchOutput


class AcademicSearchAgent:
    """Competition-oriented academic paper search agent."""

    def __init__(
        self,
        config: AppConfig,
        use_llm: bool = True,
        force_fallback_models: bool = False,
    ):
        self.config = config
        self.llm_client: Optional[LLMClient] = LLMClient(config.llm) if use_llm else None
        self.planner = LLMPlanner(self.llm_client)
        self.evolver = LLMQueryEvolver(self.llm_client)
        self.verifier = LLMVerifier(self.llm_client)
        self.synthesizer = ResultSynthesizer(self.llm_client)
        self.retriever = AcademicRetriever(config.retrieval)
        self.ranker = CompetitionRanker(
            config.ranking,
            config.small_models,
            force_fallback_models=force_fallback_models,
        )

    def search(self, query: str, top_k: int = 20, synthesize: bool = True) -> SearchOutput:
        started = time.time()
        self.retriever.reset_stats()
        if self.llm_client:
            self.llm_client.reset_stats()

        trace: List[AgentTrace] = []
        plan = self.planner.plan(query)
        self._add_trace(
            trace,
            role="Planner",
            action="multi-dimensional query parsing",
            detail=(
                f"intent={plan.intent or query}; entities={len(plan.entities)}; "
                f"methods={len(plan.methods)}; datasets={len(plan.datasets)}; "
                f"constraints={len(plan.constraints)}"
            ),
            queries=plan.sub_queries,
        )
        queries = list(dict.fromkeys(plan.sub_queries or [query]))
        scoring_query = self._scoring_query(query, plan)
        strategies = self._initial_strategies(query, plan, scoring_query)

        candidates: List[Paper] = []
        for round_id in range(max(1, self.config.retrieval.max_rounds)):
            if self.retriever.api_calls >= self.config.budget.max_api_calls_per_query:
                break
            if round_id == 0:
                active_strategies = strategies
            else:
                evolved = self._next_queries(
                    query,
                    scoring_query,
                    candidates,
                    existing=queries,
                    use_llm=synthesize,
                )
                active_strategies = [
                    {
                        "name": "query-evolution",
                        "detail": "Crawler adjusts search terms from high-scoring papers.",
                        "queries": evolved,
                    }
                ]
                queries = list(dict.fromkeys(queries + evolved))

            for strategy in active_strategies:
                all_strategy_queries = _unique([q for q in strategy["queries"] if q])
                if not all_strategy_queries:
                    continue
                before = len(candidates)
                strategy_queries = self._budgeted_queries(all_strategy_queries)
                if not strategy_queries:
                    continue
                found = self.retriever.search_many(strategy_queries)
                candidates = deduplicate(candidates + found)
                candidates = self.ranker.rank(scoring_query, candidates)[: self.config.retrieval.max_candidates]
                self._add_trace(
                    trace,
                    role="Crawler",
                    action=f"round {round_id + 1}: {strategy['name']}",
                    detail=strategy["detail"],
                    queries=all_strategy_queries,
                    candidates_before=before,
                    candidates_after=len(candidates),
                    selected_count=len(found),
                )
                if self.retriever.api_calls >= self.config.budget.max_api_calls_per_query:
                    break
            candidates = self.ranker.rank(scoring_query, candidates)[: self.config.retrieval.max_candidates]
            if (
                self.config.retrieval.citation_expand_limit > 0
                and candidates
                and self.retriever.api_calls < self.config.budget.max_api_calls_per_query
            ):
                before = len(candidates)
                expanded = self.retriever.expand_citation_network(
                    candidates[: self.config.retrieval.citation_expand_seeds],
                    max_api_calls=self.config.budget.max_api_calls_per_query,
                )
                if expanded:
                    candidates = deduplicate(candidates + expanded)
                    candidates = self.ranker.rank(scoring_query, candidates)[: self.config.retrieval.max_candidates]
                self._add_trace(
                    trace,
                    role="Crawler",
                    action=f"round {round_id + 1}: citation-network expansion",
                    detail="Follow one-hop references/citations from high-score seeds to improve coverage.",
                    candidates_before=before,
                    candidates_after=len(candidates),
                    selected_count=len(expanded),
                )
            if round_id + 1 >= self.config.retrieval.max_rounds or not candidates:
                break

        candidates = candidates[: self.config.retrieval.max_candidates]
        verify_n = min(self.config.ranking.llm_verify_top_n, len(candidates))
        if self.llm_client and self.llm_client.calls < self.config.budget.max_llm_calls_per_query:
            selector_candidates = self._selector_candidates(candidates, verify_n)
            self._add_trace(
                trace,
                role="Selector",
                action="preselect verification queue",
                detail="Selector samples high-score and diverse candidates for LLM relevance judgment.",
                candidates_before=len(candidates),
                candidates_after=len(selector_candidates),
                selected_count=len(selector_candidates),
            )
            batch_size = max(1, self.config.ranking.llm_verifier_batch_size)
            for batch in _chunks(selector_candidates, batch_size):
                # Reserve one LLM call for final result synthesis whenever possible.
                reserve_calls = 1 if synthesize else 0
                if self.llm_client.calls >= max(1, self.config.budget.max_llm_calls_per_query - reserve_calls):
                    break
                self.verifier.verify(query, batch)
                self._add_trace(
                    trace,
                    role="Selector",
                    action="batch relevance verification",
                    detail="LLM verifier labels high/partial/irrelevant candidates and assigns a fine-grained score.",
                    candidates_before=len(batch),
                    candidates_after=len([p for p in batch if p.llm_score > 0]),
                    selected_count=len(batch),
                )
            candidates = self.ranker.rank(scoring_query, candidates)

        before_filter = len(candidates)
        candidates = self._selector_filter(candidates, top_k)
        self._add_trace(
            trace,
            role="Selector",
            action="noise filtering",
            detail="Remove obvious irrelevant papers while preserving enough recall for final ranking.",
            candidates_before=before_filter,
            candidates_after=len(candidates),
            selected_count=len(candidates),
        )
        candidates = self._filter_textually_related(candidates)
        if not synthesize:
            candidates = self._selector_first_sort(scoring_query, candidates)
        self._add_trace(
            trace,
            role="Ranker",
            action="authority-recency-diversity ranking",
            detail="Fuse API, BM25, Embedding, Reranker, LLM, authority, recency and diversity signals.",
            candidates_before=before_filter,
            candidates_after=len(candidates),
            selected_count=min(top_k, len(candidates)),
        )

        top_papers = candidates[:top_k]
        summary = self._summary(query, top_papers)
        synthesis = self.synthesizer.synthesize(query, top_papers) if synthesize else {}
        stats = AgentStats(
            llm_calls=self.llm_client.calls if self.llm_client else 0,
            api_calls=self.retriever.api_calls,
            estimated_prompt_tokens=self.llm_client.prompt_tokens if self.llm_client else 0,
            estimated_completion_tokens=self.llm_client.completion_tokens if self.llm_client else 0,
            latency_seconds=time.time() - started,
            warnings=list(self.retriever.warnings)
            + (list(self.llm_client.warnings) if self.llm_client else []),
        )
        return SearchOutput(
            query=query,
            plan=plan,
            papers=top_papers,
            stats=stats,
            summary=summary,
            synthesis=synthesis,
            agent_trace=trace,
        )

    def _scoring_query(self, query: str, plan: QueryPlan) -> str:
        parts = [query, plan.intent]
        parts.extend(plan.entities)
        parts.extend(plan.methods)
        parts.extend(plan.datasets)
        parts.extend(plan.sub_queries)
        return " ".join(x for x in dict.fromkeys(parts) if x).strip()

    def _initial_strategies(self, query: str, plan: QueryPlan, scoring_query: str) -> List[Dict[str, List[str] | str]]:
        variants = _planner_query_variants(query, plan, scoring_query)
        core = variants[:2]
        focused_parts = plan.entities[:4] + plan.methods[:4] + plan.datasets[:4] + _constraint_terms(plan.constraints)
        focused_query = " ".join(_unique(focused_parts)) or scoring_query
        focused = _unique([focused_query, *variants[2:4]])[:2]
        strategies: List[Dict[str, List[str] | str]] = [
            {
                "name": "semantic-core",
                "detail": "High-recall semantic queries from planner decomposition.",
                "queries": core,
            },
            {
                "name": "method-dataset-focused",
                "detail": "Queries emphasize methods, datasets, venues, time and domain constraints.",
                "queries": focused,
            },
            {
                "name": "title-like",
                "detail": "A compact title-shaped query improves local title and arXiv recall.",
                "queries": variants[4:5],
            },
        ]
        if _wants_recent(query, plan):
            strategies.append(
                {
                    "name": "recency-oriented",
                    "detail": "Queries bias toward recent and state-of-the-art papers.",
                    "queries": _unique([f"{scoring_query} recent state of the art 2024 2025"]),
                }
            )
        return strategies

    def _filter_textually_related(self, candidates: List[Paper]) -> List[Paper]:
        """Drop API-only hits that have no title/abstract evidence.

        OpenAlex/Semantic Scholar sometimes return noisy results for short or
        Chinese queries. If our local text signals are all zero, showing the
        paper is more misleading than returning fewer results.
        """

        if not candidates:
            return []
        recall_sources = {"SerperArxiv", "arXiv", "PaSaTitleDB"}
        filtered = []
        for p in candidates:
            source = _source_family(p.source)
            text_signal = p.bm25_score + p.embedding_score + p.reranker_score
            if text_signal > 1e-9 or p.llm_score >= 0.25:
                filtered.append(p)
                continue
            if source in recall_sources and p.title:
                filtered.append(p)
                continue
            if "arxiv" in (p.doi + " " + p.url + " " + p.paper_id).lower() and p.title:
                filtered.append(p)
        return filtered

    def _selector_first_sort(self, query: str, candidates: List[Paper]) -> List[Paper]:
        """PaSa-style formal-evaluation ordering.

        PaSa's reported recall@20/50/100 sorts crawled papers by selector
        confidence.  In this lightweight system not every candidate can be sent
        to the LLM, so the final score blends LLM confidence with the neural
        ranker and source/search-engine score instead of burying all unverified
        candidates.
        """

        def score(p: Paper):
            label = (p.relevance_label or "").lower()
            label_bonus = 0.0
            if label.startswith("high"):
                label_bonus = 0.35
            elif label.startswith("partial"):
                label_bonus = 0.12
            elif label.startswith("irrelevant"):
                label_bonus = -0.35
            source_bonus = 0.04 if _source_family(p.source) in {"SerperArxiv", "arXiv", "PaSaTitleDB"} else 0.0
            title_signal = _title_query_score(query, p)
            sparse_penalty = -0.025 if not (p.abstract or p.venue or p.doi) and _source_family(p.source) not in {"PaSaTitleDB"} else 0.0
            score_value = (
                p.llm_score
                + label_bonus
                + source_bonus
                + 0.32 * p.final_score
                + 0.10 * p.api_score
                + 0.08 * p.reranker_score
                + 0.05 * p.embedding_score
                + 0.12 * title_signal
                + 0.02 * p.authority_score
                + sparse_penalty
            )
            return (score_value, p.llm_score, title_signal, p.final_score, p.api_score)

        return sorted(candidates, key=score, reverse=True)

    def _selector_candidates(self, candidates: List[Paper], limit: int) -> List[Paper]:
        if limit <= 0:
            return []
        selected: List[Paper] = []
        buckets: Dict[str, List[Paper]] = {}
        for paper in candidates:
            source = _source_family(paper.source)
            buckets.setdefault(source, []).append(paper)

        # Give each retrieval source a small verifier budget.  This keeps
        # high-recall sources such as arXiv/PaSaTitleDB from being crowded out
        # by OpenAlex/Semantic Scholar candidates before the LLM can judge them.
        per_source = max(1, limit // max(1, len(buckets)))
        priority_sources = ["SerperArxiv", "arXiv", "PaSaTitleDB", "SemanticScholar", "OpenAlex"]
        ordered_sources = priority_sources + [s for s in buckets if s not in priority_sources]
        for source in ordered_sources:
            for p in buckets.get(source, [])[:per_source]:
                selected.append(p)
            if len(selected) >= max(2, limit // 2):
                break
        for p in candidates:
            if p not in selected:
                selected.append(p)
            if len(selected) >= limit:
                break
        return selected

    def _selector_filter(self, candidates: List[Paper], top_k: int) -> List[Paper]:
        if not candidates:
            return []
        filtered = []
        for p in candidates:
            label = (p.relevance_label or "").lower()
            source = _source_family(p.source)
            recall_source = source in {"SerperArxiv", "arXiv", "PaSaTitleDB"}
            if label.startswith("irrelevant") and len(candidates) > top_k and not recall_source:
                continue
            if 0.0 < p.llm_score < 0.12 and len(candidates) > top_k * 2 and not recall_source:
                continue
            filtered.append(p)
        return filtered or candidates

    def _budgeted_queries(self, queries: List[str]) -> List[str]:
        remaining = self.config.budget.max_api_calls_per_query - self.retriever.api_calls
        if remaining <= 0:
            return []
        selected: List[str] = []
        spent = 0
        serper_left = max(
            0,
            self.config.retrieval.serper_query_limit
            - getattr(self.retriever, "_serper_queries_used", 0),
        )
        arxiv_left = max(
            0,
            self.config.retrieval.arxiv_query_limit
            - getattr(self.retriever, "_arxiv_queries_used", 0),
        )
        for q in _unique([q for q in queries if q]):
            cost, uses_serper, uses_arxiv = self._estimated_query_cost(serper_left, arxiv_left)
            if spent + cost > remaining:
                break
            selected.append(q)
            spent += cost
            if uses_serper:
                serper_left -= 1
            if uses_arxiv:
                arxiv_left -= 1
        return selected

    def _estimated_query_cost(self, serper_left: int, arxiv_left: int) -> tuple[int, bool, bool]:
        cost = int(self.config.retrieval.use_openalex) + int(self.config.retrieval.use_semantic_scholar)
        uses_serper = (
            self.config.retrieval.use_serper
            and bool(self.config.retrieval.serper_api_key)
            and serper_left > 0
            and self.config.retrieval.serper_query_variants > 0
        )
        uses_arxiv = (
            self.config.retrieval.use_arxiv
            and arxiv_left > 0
            and self.config.retrieval.arxiv_query_variants > 0
        )
        if uses_serper:
            # Each Serper variant is one web-search call; one extra arXiv call
            # fetches metadata for the extracted ids.
            cost += min(3, max(0, self.config.retrieval.serper_query_variants)) + 1
        if uses_arxiv:
            cost += min(3, max(0, self.config.retrieval.arxiv_query_variants))
        return max(0, cost), uses_serper, uses_arxiv

    def _add_trace(
        self,
        trace: List[AgentTrace],
        *,
        role: str,
        action: str,
        detail: str = "",
        queries: Optional[List[str]] = None,
        candidates_before: int = 0,
        candidates_after: int = 0,
        selected_count: int = 0,
    ) -> None:
        trace.append(
            AgentTrace(
                step=len(trace) + 1,
                role=role,
                action=action,
                detail=detail,
                queries=list(queries or [])[:8],
                candidates_before=candidates_before,
                candidates_after=candidates_after,
                selected_count=selected_count,
            )
        )

    def _next_queries(
        self,
        original_query: str,
        scoring_query: str,
        candidates: List[Paper],
        existing: List[str],
        use_llm: bool = True,
    ) -> List[str]:
        if not candidates:
            return []
        if use_llm:
            evolved = self.evolver.evolve(original_query, scoring_query, candidates, existing)
            if evolved:
                return evolved
        top_text = " ".join((p.title + " " + p.abstract[:500] + " " + p.venue) for p in candidates[:10])
        banned = set(_tokens(scoring_query)) | {
            "article",
            "paper",
            "review",
            "study",
            "using",
            "based",
            "approach",
            "result",
            "results",
            "method",
            "methods",
        }
        terms = [t for t, _ in Counter(_tokens(top_text)).most_common(30) if t not in banned]
        new_queries = []
        if terms:
            new_queries.append(scoring_query + " " + " ".join(terms[:5]))
        if len(terms) >= 8:
            new_queries.append(" ".join(_tokens(scoring_query)[:6] + terms[5:10]))
        if len(terms) >= 14:
            new_queries.append(" ".join(terms[8:14]))
        return [q for q in dict.fromkeys(new_queries) if q and q not in existing]

    def _summary(self, query: str, papers: List[Paper]) -> str:
        if not papers:
            return "No relevant papers found."
        high = [p for p in papers if p.final_score >= papers[0].final_score * 0.75]
        return (
            f"Query: {query}\n"
            f"Returned {len(papers)} papers. {len(high)} papers are high-confidence candidates. "
            f"Top result: {papers[0].title}."
        )


def _tokens(text: str):
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())


def _unique(items: List[str]) -> List[str]:
    return [x for x in dict.fromkeys(str(i).strip() for i in items if str(i).strip()) if x]


def _source_family(source: str) -> str:
    source = source or "unknown"
    for family in ["SerperArxiv", "arXiv", "PaSaTitleDB", "SemanticScholar", "OpenAlex"]:
        if family in source:
            return family
    return source


def _planner_query_variants(query: str, plan: QueryPlan, scoring_query: str) -> List[str]:
    """Build at most five stable query variants for formal evaluation.

    The LLM planner may return many broad rewrites.  The retriever is more
    stable when it receives a compact set that covers semantic, method/dataset,
    constraint and title-like intents without multiplying API calls.
    """

    focused_parts = plan.entities[:5] + plan.methods[:5] + plan.datasets[:5] + _constraint_terms(plan.constraints)
    focused = " ".join(_unique(focused_parts))
    title_like = _title_like_query(query, plan, scoring_query)
    variants = _unique(
        [
            query,
            plan.intent,
            *(plan.sub_queries or [])[:2],
            focused,
            title_like,
        ]
    )
    return variants[:5] or [query]


def _title_like_query(query: str, plan: QueryPlan, scoring_query: str) -> str:
    pieces: List[str] = []
    pieces.extend(plan.methods[:4])
    pieces.extend(plan.datasets[:4])
    pieces.extend(plan.entities[:6])
    if not pieces:
        pieces = _tokens(scoring_query or query)[:12]
    text = " ".join(_unique(pieces))
    if text:
        return text
    return " ".join(_tokens(query)[:12])


def _title_query_score(query: str, paper: Paper) -> float:
    query_terms = [t for t in _tokens(query) if t not in _RANK_STOPWORDS]
    title_terms = [t for t in _tokens(paper.title) if t not in _RANK_STOPWORDS]
    if not query_terms or not title_terms:
        return 0.0
    qset = set(query_terms)
    tset = set(title_terms)
    coverage = len(qset & tset) / max(1, min(len(qset), 12))
    ordered_hits = 0
    title_text = " ".join(title_terms)
    for n in (4, 3, 2):
        for i in range(0, max(0, len(query_terms) - n + 1)):
            phrase = " ".join(query_terms[i : i + n])
            if phrase and phrase in title_text:
                ordered_hits += n
                break
        if ordered_hits:
            break
    phrase_bonus = min(0.25, ordered_hits / 16.0)
    return min(1.0, 0.75 * coverage + phrase_bonus)


def _constraint_terms(constraints) -> List[str]:
    if not constraints:
        return []
    if isinstance(constraints, dict):
        terms = []
        for key, value in constraints.items():
            if isinstance(value, list):
                terms.extend(str(x) for x in value)
            elif value:
                terms.append(f"{key} {value}")
        return terms[:8]
    if isinstance(constraints, list):
        return [str(x) for x in constraints[:8]]
    return [str(constraints)]


def _wants_recent(query: str, plan: QueryPlan) -> bool:
    text = " ".join([query, str(plan.constraints), " ".join(plan.sub_queries)]).lower()
    return any(x in text for x in ["recent", "latest", "sota", "state-of-the-art", "2024", "2025", "2026", "最新", "近年", "近年来"])


def _chunks(items: List[Paper], size: int):
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]


_RANK_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "between",
    "from",
    "give",
    "into",
    "large",
    "model",
    "models",
    "paper",
    "papers",
    "result",
    "results",
    "show",
    "study",
    "that",
    "the",
    "this",
    "using",
    "which",
    "with",
}
