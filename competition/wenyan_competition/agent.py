from __future__ import annotations

import re
import time
from collections import Counter
from typing import Dict, List, Optional

from .config import AppConfig
from .constraints import apply_constraint_policy, build_constraint_coverage, constraint_gap_queries
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
        include_title_sources = not (
            self.config.retrieval.suppress_title_only_for_metadata_queries
            and _requires_verifiable_metadata(query, plan)
        )

        candidates: List[Paper] = self.retriever.search_exact_bibliographic(
            query,
            plan.sub_queries or [plan.intent],
        )
        if candidates:
            self._add_trace(
                trace,
                role="Crawler",
                action="exact bibliographic lookup",
                detail="Resolve an explicitly named title/author/citation-key before topical expansion.",
                queries=[query],
                candidates_before=0,
                candidates_after=len(candidates),
                selected_count=len(candidates),
            )
        metadata_candidates = self.retriever.search_openalex_metadata_constraints(
            query,
            plan.sub_queries,
        )
        if metadata_candidates:
            before_metadata = len(candidates)
            candidates = deduplicate([*candidates, *metadata_candidates])
            self._add_trace(
                trace,
                role="Crawler",
                action="structured author metadata lookup",
                detail="Resolve author identities first, then retrieve works under explicit year/publisher constraints.",
                queries=plan.sub_queries,
                candidates_before=before_metadata,
                candidates_after=len(candidates),
                selected_count=len(metadata_candidates),
            )
        metadata_direct = bool(
            metadata_candidates
            and self.config.retrieval.metadata_direct_shortcut
            and _requires_verifiable_metadata(query, plan)
        )
        if metadata_direct:
            candidates = self.ranker.rank(scoring_query, candidates)
            self._add_trace(
                trace,
                role="BudgetController",
                action="metadata direct shortcut",
                detail=(
                    "Structured lookup returned verifiable candidates; skip broad "
                    "title retrieval and rank the evidence-bearing set directly."
                ),
                candidates_before=len(candidates),
                candidates_after=len(candidates),
                selected_count=len(candidates),
            )
        retrieval_rounds = 0
        stopped_early = False
        stop_reason = ""
        round_limit = 0 if metadata_direct else max(1, self.config.retrieval.max_rounds)
        for round_id in range(round_limit):
            if self.retriever.api_calls >= self.config.budget.max_api_calls_per_query:
                stop_reason = "API budget exhausted"
                stopped_early = True
                break
            round_before_keys = {paper.key() for paper in candidates}
            if round_id == 0:
                active_strategies = strategies
            else:
                gap_queries = []
                if self.config.retrieval.use_gap_driven_evolution:
                    gap_queries = constraint_gap_queries(
                        query,
                        plan,
                        candidates[: max(top_k * 2, 30)],
                        max_queries=self.config.retrieval.gap_query_limit,
                        min_coverage=self.config.retrieval.gap_min_coverage,
                    )
                evolved = gap_queries or self._next_queries(
                    query,
                    scoring_query,
                    candidates,
                    existing=queries,
                    use_llm=synthesize,
                )
                if not evolved:
                    stopped_early = True
                    stop_reason = "no useful second-round query"
                    self._add_trace(
                        trace,
                        role="BudgetController",
                        action="stop retrieval",
                        detail=stop_reason,
                        candidates_before=len(candidates),
                        candidates_after=len(candidates),
                    )
                    break
                active_strategies = [
                    {
                        "name": "coverage-gap evolution" if gap_queries else "query-evolution",
                        "detail": (
                            "Crawler targets query dimensions not covered by first-round candidates."
                            if gap_queries
                            else "Crawler adjusts search terms from high-scoring papers."
                        ),
                        "queries": evolved,
                    }
                ]
                queries = list(dict.fromkeys(queries + evolved))

            for strategy in active_strategies:
                reserve = (
                    self.config.retrieval.gap_api_reserve
                    if round_id == 0
                    and self.config.retrieval.use_gap_driven_evolution
                    and self.config.retrieval.max_rounds > 1
                    else 0
                )
                strategy_queries = self._budgeted_queries(strategy["queries"], reserve=reserve)
                if not strategy_queries:
                    continue
                before = len(candidates)
                found = self.retriever.search_many(
                    strategy_queries,
                    include_title_sources=include_title_sources,
                )
                candidates = deduplicate(candidates + found)
                candidates = self.ranker.rank(scoring_query, candidates)[: self.config.retrieval.max_candidates]
                self._add_trace(
                    trace,
                    role="Crawler",
                    action=f"round {round_id + 1}: {strategy['name']}",
                    detail=strategy["detail"],
                    queries=strategy_queries,
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
                and (
                    round_id + 1 >= self.config.retrieval.max_rounds
                    or not self.config.retrieval.use_gap_driven_evolution
                )
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
                retrieval_rounds = round_id + 1
                break
            retrieval_rounds = round_id + 1
            new_unique = len({paper.key() for paper in candidates} - round_before_keys)
            round_coverage = build_constraint_coverage(
                query,
                plan,
                candidates[: max(top_k * 2, 30)],
            )
            stop_reason = self._retrieval_stop_reason(
                started=started,
                candidates=candidates,
                new_unique=new_unique,
                coverage=round_coverage,
                top_k=top_k,
            )
            if stop_reason:
                stopped_early = True
                self._add_trace(
                    trace,
                    role="BudgetController",
                    action="adaptive early stop",
                    detail=stop_reason,
                    candidates_before=len(candidates),
                    candidates_after=len(candidates),
                    selected_count=min(top_k, len(candidates)),
                )
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
            candidates = self._selector_first_sort(candidates)
        before_constraints = len(candidates)
        candidates = apply_constraint_policy(
            query,
            plan,
            candidates,
            top_k,
            weight=self.config.ranking.constraint_weight,
            hard_filter_year=self.config.ranking.constraint_hard_filter_year,
        )
        before_dense_head = [paper.key() for paper in candidates[: self.config.ranking.dense_head_size]]
        candidates = protect_dense_head(candidates, self.config.ranking)
        after_dense_head = [paper.key() for paper in candidates[: self.config.ranking.dense_head_size]]
        if before_dense_head != after_dense_head:
            dense_only_count = sum(
                1
                for paper in candidates[: self.config.ranking.dense_head_size]
                if is_dense_only_candidate(paper)
            )
            self._add_trace(
                trace,
                role="Ranker",
                action="dense head admission",
                detail=(
                    "Protect Top20 precision with embedding/reranker admission; "
                    f"dense-only admitted={dense_only_count}. Deferred candidates remain in Top21-100."
                ),
                candidates_before=len(candidates),
                candidates_after=len(candidates),
                selected_count=min(self.config.ranking.dense_head_size, len(candidates)),
            )
        constraint_coverage = build_constraint_coverage(query, plan, candidates[:top_k])
        if constraint_coverage["total_dimensions"]:
            self._add_trace(
                trace,
                role="ConstraintEngine",
                action="explicit constraint execution",
                detail=(
                    f"Evaluate year, venue, publication type, method, dataset and topic evidence; "
                    f"covered={constraint_coverage['covered_dimensions']}/"
                    f"{constraint_coverage['total_dimensions']}."
                ),
                candidates_before=before_constraints,
                candidates_after=len(candidates),
                selected_count=min(top_k, len(candidates)),
            )
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
            cache_hits=self.retriever.cache_hits,
            cache_misses=self.retriever.cache_misses,
            retrieval_rounds=retrieval_rounds,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
        return SearchOutput(
            query=query,
            plan=plan,
            papers=top_papers,
            stats=stats,
            summary=summary,
            synthesis=synthesis,
            agent_trace=trace,
            constraint_coverage=constraint_coverage,
        )


    def _scoring_query(self, query: str, plan: QueryPlan) -> str:
        parts = [query, plan.intent]
        parts.extend(plan.entities)
        parts.extend(plan.methods)
        parts.extend(plan.datasets)
        parts.extend(plan.sub_queries)
        return " ".join(x for x in dict.fromkeys(parts) if x).strip()

    def _initial_strategies(self, query: str, plan: QueryPlan, scoring_query: str) -> List[Dict[str, List[str] | str]]:
        core = _unique((plan.sub_queries or [query])[:4])
        focused_parts = plan.entities[:4] + plan.methods[:4] + plan.datasets[:4] + _constraint_terms(plan.constraints)
        focused_query = " ".join(_unique(focused_parts)) or scoring_query
        strategies: List[Dict[str, List[str] | str]] = [
            {
                "name": "semantic-core",
                "detail": "High-recall semantic queries from planner decomposition.",
                "queries": core,
            },
            {
                "name": "constraint-focused",
                "detail": "Queries emphasize methods, datasets, venues, time and domain constraints.",
                "queries": _unique([focused_query, f"{scoring_query} {focused_query}"])[:2],
            },
            {
                "name": "authority-oriented",
                "detail": "Queries bias toward survey, benchmark and influential scholarly papers.",
                "queries": _unique([
                    f"{scoring_query} survey review benchmark",
                    f"{scoring_query} influential highly cited",
                ]),
            },
        ]
        if _wants_recent(query, plan):
            strategies.append(
                {
                    "name": "recency-oriented",
                    "detail": "Queries bias toward recent and state-of-the-art papers.",
                    "queries": _unique([
                        f"{scoring_query} recent advances state of the art",
                        f"{scoring_query} 2024 2025 2026",
                    ]),
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
        filtered = [
            p
            for p in candidates
            if (p.bm25_score + p.embedding_score + p.reranker_score) > 1e-9 or p.llm_score >= 0.25
        ]
        return filtered

    def _selector_first_sort(self, candidates: List[Paper]) -> List[Paper]:
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
            family = _source_family(p.source)
            source_bonus = 0.18 if family in {"SemanticScholarExact", "OpenAlexMetadata", "CrossrefMetadata"} else (
                0.025 if family in {"SerperArxiv", "arXiv", "PaSaTitleDB"} else 0.0
            )
            score_value = (
                p.llm_score
                + label_bonus
                + source_bonus
                + 0.32 * p.final_score
                + 0.10 * p.api_score
                + 0.08 * p.reranker_score
                + 0.05 * p.embedding_score
            )
            return (score_value, p.llm_score, p.final_score, p.api_score)

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
        priority_sources = [
            "SemanticScholarExact",
            "OpenAlexMetadata",
            "CrossrefMetadata",
            "SerperArxiv",
            "arXiv",
            "PaSaTitleDB",
            "SemanticScholar",
            "OpenAlex",
        ]
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
            if label.startswith("irrelevant") and len(candidates) > top_k:
                continue
            if 0.0 < p.llm_score < 0.12 and len(candidates) > top_k * 2:
                continue
            filtered.append(p)
        return filtered or candidates

    def _budgeted_queries(self, queries: List[str], reserve: int = 0) -> List[str]:
        gross_remaining = self.config.budget.max_api_calls_per_query - self.retriever.api_calls
        if gross_remaining < 0:
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
        unique_queries = _unique([q for q in queries if q])
        if not unique_queries:
            return []
        first_cost, _, _ = self._estimated_query_cost(serper_left, arxiv_left)
        # Never reserve so much that the current round cannot execute one
        # query. A local-only query has zero API cost and remains executable.
        effective_reserve = min(max(0, reserve), max(0, gross_remaining - first_cost))
        remaining = gross_remaining - effective_reserve
        for q in unique_queries:
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

    def _retrieval_stop_reason(
        self,
        *,
        started: float,
        candidates: List[Paper],
        new_unique: int,
        coverage: Dict,
        top_k: int,
    ) -> str:
        retrieval = self.config.retrieval
        elapsed = time.time() - started
        latency_limit = max(1.0, float(self.config.budget.max_latency_seconds))
        if elapsed >= latency_limit * max(0.1, retrieval.early_stop_budget_fraction):
            return "latency reserve reached; preserve time for verification and synthesis"
        if not retrieval.early_stop_enabled:
            return ""
        candidate_target = min(
            retrieval.max_candidates,
            max(top_k * 2, retrieval.early_stop_min_candidates),
        )
        enough_candidates = len(candidates) >= candidate_target
        dimensions_covered = (
            coverage.get("total_dimensions", 0) > 0
            and coverage.get("covered_dimensions", 0) == coverage.get("total_dimensions", 0)
            and coverage.get("overall_coverage", 0.0) >= retrieval.early_stop_coverage
        )
        if enough_candidates and dimensions_covered:
            return "candidate pool is sufficient and all query dimensions are covered"
        if enough_candidates and new_unique == 0:
            return "previous round added no new canonical papers"
        return ""

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


def protect_dense_head(papers: List[Paper], ranking_config) -> List[Paper]:
    """Limit unsupported dense-only papers in the head without deleting them."""

    if not ranking_config.dense_head_protection or not papers:
        return papers
    head_size = min(max(1, ranking_config.dense_head_size), len(papers))
    dense_quota = max(0, ranking_config.dense_head_max_dense_only)
    selected_indexes: List[int] = []
    admitted_dense = 0

    for index, paper in enumerate(papers):
        if len(selected_indexes) >= head_size:
            break
        if not is_dense_only_candidate(paper):
            selected_indexes.append(index)
            continue
        model_supported = (
            paper.embedding_score >= ranking_config.dense_head_min_embedding
            and paper.reranker_score >= ranking_config.dense_head_min_reranker
        )
        llm_supported = paper.llm_score >= 0.80 or paper.relevance_label.lower().startswith("high")
        if admitted_dense < dense_quota and (model_supported or llm_supported):
            selected_indexes.append(index)
            admitted_dense += 1

    if len(selected_indexes) < head_size:
        selected = set(selected_indexes)
        for index in range(len(papers)):
            if index not in selected:
                selected_indexes.append(index)
                selected.add(index)
                if len(selected_indexes) >= head_size:
                    break

    selected = set(selected_indexes)
    return [papers[index] for index in selected_indexes] + [
        paper for index, paper in enumerate(papers) if index not in selected
    ]


def is_dense_only_candidate(paper: Paper) -> bool:
    sources = {source for source in (paper.source or "").split("+") if source}
    return sources == {"DenseTitleDB"}


def _tokens(text: str):
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())


def _unique(items: List[str]) -> List[str]:
    return [x for x in dict.fromkeys(str(i).strip() for i in items if str(i).strip()) if x]


def _source_family(source: str) -> str:
    source = source or "unknown"
    for family in ["SemanticScholarExact", "OpenAlexMetadata", "CrossrefMetadata", "SerperArxiv", "arXiv", "PaSaTitleDB", "SemanticScholar", "OpenAlex"]:
        if family in source:
            return family
    return source


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


def _requires_verifiable_metadata(query: str, plan: QueryPlan) -> bool:
    """Detect constraints that cannot be checked from title text alone."""

    text = " ".join([query, str(plan.constraints)]).lower()
    explicit_patterns = [
        r"\bpapers?\s+(?:written|authored|co-authored)\s+by\b",
        r"\bpapers?\s+by\b",
        r"\bpapers?\s+(?:that\s+)?cit(?:e|es|ed|ing)\b",
        r"\b(?:more|fewer|less)\s+than\s+\d+\s+citations?\b",
        r"\bat\s+least\s+\d+\s+(?:citations?|authors?)\b",
        r"\bcited\s+by\s+at\s+least\b",
        r"\bpublished\s+(?:at|in|by)\b",
    ]
    if any(re.search(pattern, text) for pattern in explicit_patterns):
        return True

    keys = {str(key).lower() for key in (plan.constraints or {})}
    metadata_keys = {
        "author", "authors", "coauthor", "venue", "conference", "journal",
        "publisher", "cites", "cited_by", "min_citations", "citation_count",
        "min_authors",
    }
    return bool(keys & metadata_keys)


def _wants_recent(query: str, plan: QueryPlan) -> bool:
    text = " ".join([query, str(plan.constraints), " ".join(plan.sub_queries)]).lower()
    return any(x in text for x in ["recent", "latest", "sota", "state-of-the-art", "2024", "2025", "2026", "最新", "近年", "近年来"])


def _chunks(items: List[Paper], size: int):
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]
