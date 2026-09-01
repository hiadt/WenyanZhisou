from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import requests

from wenyan_competition.agent import (
    _requires_verifiable_metadata,
    is_dense_only_candidate,
    merge_baseline_head,
)
from wenyan_competition.config import RetrievalConfig, load_config
from wenyan_competition.constraints import (
    apply_constraint_policy,
    build_constraint_coverage,
    constraint_gap_queries,
)
from wenyan_competition.dataset import extract_gold_items
from wenyan_competition.llm import _as_dict, _as_list, heuristic_plan, heuristic_synthesis
from wenyan_competition.retrievers import (
    AcademicRetriever,
    _arxiv_id_from_url,
    _arxiv_queries,
    _extract_arxiv_ids_from_serper,
    _author_names_from_queries,
    _looks_like_author_metadata_lookup,
    _looks_like_bibliographic_lookup,
    _openalex_api_work_url,
    _serper_arxiv_queries,
    deduplicate,
    fuse_title_results,
)
from wenyan_competition.schema import Paper, QueryPlan
from evaluate_pasa import _apply_formal_eval_defaults, flexible_recall_at, paper_aliases
from train_selector_reranker import load_selector_jsonl, parse_selector_record
from benchmark_dense_title_retrieval import (
    normalize_arxiv_id,
    quality_gate,
    recall_metrics,
    reciprocal_rank_fusion,
)
from evaluate_asta_paper_finder import (
    AstaSample,
    build_asta_output,
    diagnostic_metrics,
    direct_corpus_id,
    extract_known_corpus_ids,
    load_asta_samples,
    semantic_scholar_lookup_id,
)
from asta_solver_service import official_resolved_results


ROOT = Path(__file__).resolve().parent


def main() -> None:
    checks = [
        check_config_loads,
        check_chinese_medical_expansion,
        check_web_starts_blank,
        check_full_text_is_searchable_text,
        check_synthesis_shape,
        check_llm_planner_loose_fields,
        check_pasa_gold_matching,
        check_pasa_title_retriever,
        check_arxiv_query_helpers,
        check_serper_arxiv_helpers,
        check_formal_eval_defaults,
        check_openalex_url_normalization,
        check_citation_fetch_warnings_are_quiet,
        check_cross_source_identity_fusion,
        check_constraint_execution_and_coverage,
        check_gap_driven_query_evolution,
        check_normalized_api_cache,
        check_transient_retrieval_retry,
        check_selector_training_parser,
        check_dense_title_benchmark_helpers,
        check_title_rrf_fusion,
        check_dense_head_protection,
        check_asta_paper_finder_adapter,
        check_exact_bibliographic_routing,
        check_author_metadata_routing,
        check_metadata_source_hygiene_routing,
        check_asta_official_output_policy,
        check_smoke_command,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("All offline quality checks passed.")


def check_config_loads() -> None:
    for name in ["config.smoke.json", "config.online.json", "config.example.yaml"]:
        cfg = load_config(ROOT / name)
        assert cfg.retrieval.per_query > 0


def check_chinese_medical_expansion() -> None:
    plan = heuristic_plan("心脏支架")
    joined = "\n".join(plan.sub_queries).lower()
    assert "coronary stent" in joined
    assert "drug" in joined and "stent" in joined


def check_web_starts_blank() -> None:
    html = (ROOT / "web_demo.py").read_text(encoding="utf-8")
    assert "health().then(search)" not in html
    assert "placeholder=\"输入论文检索问题或关键词\"" in html
    assert "data-tab=\"trace\"" in html
    assert "data-tab=\"synthesis\"" in html
    assert "function renderTrace" in html
    assert "function renderSynthesis" in html
    assert "authority_score" in html and "recency_score" in html and "diversity_score" in html
    assert "<textarea id=\"query\">large language model" not in html


def check_full_text_is_searchable_text() -> None:
    paper = Paper(paper_id="T1", title="Short title", abstract="", full_text="full text evidence about coronary stent restenosis")
    assert "coronary stent restenosis" in paper.text()


def check_synthesis_shape() -> None:
    paper = Paper(
        paper_id="T1",
        title="Drug-Eluting Stents for Coronary Restenosis",
        abstract="A clinical paper about coronary stent restenosis and treatment outcomes.",
        final_score=0.9,
        embedding_score=0.8,
        reranker_score=0.7,
    )
    data = heuristic_synthesis("coronary stent restenosis", [paper])
    assert data["overview"]
    assert data["highly_relevant"]


def check_llm_planner_loose_fields() -> None:
    assert _as_dict(["time range: recent", "venue: ACL"]) == {
        "items": ["time range: recent", "venue: ACL"]
    }
    assert _as_dict("recent papers") == {"text": "recent papers"}
    assert _as_list({"method": "reranking"}) == ["method: reranking"]


def check_pasa_gold_matching() -> None:
    row = {
        "question": "Find data pruning papers for LLM pretraining.",
        "answer": ["When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale"],
        "answer_arxiv_id": ["2309.04564"],
    }
    gold_items = extract_gold_items(row)
    by_arxiv = Paper(
        paper_id="https://openalex.org/W1",
        title="Different title",
        doi="10.48550/arXiv.2309.04564",
    )
    by_title = Paper(
        paper_id="https://openalex.org/W2",
        title="When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale",
    )
    by_fused_source_id = Paper(
        paper_id="https://openalex.org/W3",
        title="Different merged title",
        source_ids=["arXiv:2309.04564"],
    )
    assert flexible_recall_at([paper_aliases(by_arxiv)], gold_items, 1) == 1.0
    assert flexible_recall_at([paper_aliases(by_title)], gold_items, 1) == 1.0
    assert flexible_recall_at([paper_aliases(by_fused_source_id)], gold_items, 1) == 1.0


def check_pasa_title_retriever() -> None:
    tmp = ROOT / "runs" / "tiny_id2paper.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(
        json.dumps(
            {
                "2309.04564": "When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale",
                "2402.09668": "How to Train Data-Efficient LLMs",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retriever = AcademicRetriever(
        RetrievalConfig(
            use_openalex=False,
            use_semantic_scholar=False,
            pasa_id2paper_path=str(tmp),
            pasa_title_limit=10,
        )
    )
    papers = retriever.search_many(
        ["smaller dataset in large language model pre-training can result in better models"]
    )
    assert any(p.paper_id == "2309.04564" for p in papers)


def check_arxiv_query_helpers() -> None:
    query = "using a smaller dataset in large language model pre-training can result in better models"
    queries = _arxiv_queries(query)
    assert any("data" in q and "pretraining" in q for q in queries)
    assert _arxiv_id_from_url("http://arxiv.org/abs/2309.04564v2") == "2309.04564"


def check_serper_arxiv_helpers() -> None:
    query = "using a smaller dataset in large language model pre-training can result in better models"
    queries = _serper_arxiv_queries(query)
    assert any("site:arxiv.org/abs" in q for q in queries)
    ids = _extract_arxiv_ids_from_serper(
        {
            "organic": [
                {"link": "https://arxiv.org/abs/2309.04564v2"},
                {"snippet": "Related work arXiv:2402.09668 about data-efficient LLMs."},
            ]
        }
    )
    assert ids == ["2309.04564", "2402.09668"]


def check_formal_eval_defaults() -> None:
    cfg = load_config(ROOT / "config.smoke.json")
    _apply_formal_eval_defaults(cfg, use_llm=True)
    assert cfg.retrieval.enable_api_cache is True
    assert cfg.retrieval.api_parallelism >= 10
    assert cfg.retrieval.max_candidates == 220
    assert cfg.retrieval.max_rounds == 1
    assert cfg.retrieval.citation_expand_limit == 0
    assert cfg.retrieval.use_gap_driven_evolution is False
    assert cfg.retrieval.early_stop_enabled is False
    assert cfg.budget.max_api_calls_per_query == 36
    assert cfg.budget.max_llm_calls_per_query == 4
    assert cfg.ranking.llm_verify_top_n == 60
    assert cfg.ranking.api_weight >= 0.14
    assert cfg.ranking.llm_verifier_weight >= 0.22
    assert cfg.ranking.constraint_weight == 0.0
    assert cfg.ranking.constraint_hard_filter_year is False

    v12 = load_config(ROOT / "config.v12.yaml")
    _apply_formal_eval_defaults(v12, use_llm=True)
    assert v12.retrieval.per_query == 18
    assert v12.retrieval.max_candidates == 220
    assert v12.retrieval.max_rounds == 1
    assert v12.retrieval.citation_expand_limit == 0
    assert v12.retrieval.use_arxiv is True
    assert v12.retrieval.use_serper is True
    assert v12.retrieval.pasa_title_limit == 80
    assert v12.ranking.embedding_weight == 0.25
    assert v12.ranking.reranker_weight == 0.25
    assert v12.ranking.authority_weight == 0.06
    assert v12.ranking.recency_weight == 0.03
    assert v12.ranking.llm_verify_top_n == 60
    assert v12.budget.max_api_calls_per_query == 36
    assert v12.budget.max_llm_calls_per_query == 4


def check_openalex_url_normalization() -> None:
    assert _openalex_api_work_url("https://openalex.org/W123") == "https://api.openalex.org/works/W123"
    assert _openalex_api_work_url("W456") == "https://api.openalex.org/works/W456"


def check_citation_fetch_warnings_are_quiet() -> None:
    text = (ROOT / "wenyan_competition" / "retrievers.py").read_text(encoding="utf-8")
    assert "fetch_openalex_work, paper_id, warn=False" in text
    assert "fetch_semantic_scholar_paper, paper_id, warn=False" in text


def check_cross_source_identity_fusion() -> None:
    papers = deduplicate(
        [
            Paper(
                paper_id="https://openalex.org/W1",
                title="Constraint-Aware Academic Search with Large Language Models",
                abstract="short",
                doi="10.48550/arXiv.2401.01234",
                source="OpenAlex",
                citation_count=10,
            ),
            Paper(
                paper_id="CorpusId:99",
                title="Constraint-Aware Academic Search with Large Language Models",
                abstract="A longer abstract with method, dataset and evaluation evidence.",
                url="https://arxiv.org/abs/2401.01234v2",
                source="SemanticScholar",
                citation_count=20,
            ),
        ]
    )
    assert len(papers) == 1
    assert papers[0].source == "OpenAlex+SemanticScholar"
    assert len(papers[0].source_ids) == 2
    assert papers[0].citation_count == 20
    assert papers[0].abstract.startswith("A longer abstract")


def check_constraint_execution_and_coverage() -> None:
    unconstrained = [
        Paper(paper_id="first", title="First paper", final_score=0.9),
        Paper(paper_id="second", title="Second paper", final_score=0.8),
    ]
    unchanged = apply_constraint_policy(
        "general academic search",
        QueryPlan(original_query="general academic search"),
        unconstrained,
        top_k=2,
    )
    assert [paper.paper_id for paper in unchanged] == ["first", "second"]

    plan = QueryPlan(
        original_query="Find transformer papers from 2022 to 2024 using BERT on SQuAD.",
        entities=["transformer"],
        methods=["BERT"],
        datasets=["SQuAD"],
        constraints={"year_range": "2022-2024", "venue": ["ACL"]},
    )
    papers = [
        Paper(
            paper_id="good",
            title="BERT Transformer Reasoning on SQuAD",
            abstract="An ACL study using BERT on the SQuAD dataset.",
            year=2023,
            venue="ACL",
            final_score=0.8,
        ),
        Paper(
            paper_id="old",
            title="Transformer Language Modeling",
            abstract="A general transformer paper.",
            year=2019,
            venue="NeurIPS",
            final_score=0.9,
        ),
    ]
    ranked = apply_constraint_policy(
        plan.original_query,
        plan,
        papers,
        top_k=1,
        weight=0.04,
        hard_filter_year=True,
    )
    assert [paper.paper_id for paper in ranked] == ["good"]
    assert ranked[0].constraint_score == 1.0
    coverage = build_constraint_coverage(plan.original_query, plan, ranked)
    assert coverage["total_dimensions"] == 5
    assert coverage["covered_dimensions"] == 5


def check_gap_driven_query_evolution() -> None:
    plan = QueryPlan(
        original_query="Find transformer reasoning papers using BERT on SQuAD.",
        intent="transformer reasoning",
        methods=["BERT"],
        datasets=["SQuAD"],
    )
    missing = [Paper(paper_id="P1", title="General Transformer Reasoning", abstract="No benchmark details.")]
    queries = constraint_gap_queries(plan.original_query, plan, missing, max_queries=2)
    assert any("squad" in query.lower() for query in queries)
    assert any("bert" in query.lower() for query in queries)

    covered = [
        Paper(
            paper_id="P2",
            title="BERT Transformer Reasoning on SQuAD",
            abstract="BERT is evaluated on the SQuAD benchmark.",
        )
    ]
    assert constraint_gap_queries(plan.original_query, plan, covered, max_queries=2) == []


def check_normalized_api_cache() -> None:
    retriever = AcademicRetriever(
        RetrievalConfig(
            use_openalex=False,
            use_semantic_scholar=False,
            use_arxiv=False,
            use_serper=False,
            pasa_id2paper_path="",
        )
    )
    calls = []

    def fake_search(query: str):
        calls.append(query)
        return [Paper(paper_id="cached", title="Cached paper")]

    retriever._cached_safe(fake_search, "  Academic   Search ")
    retriever._cached_safe(fake_search, "academic search")
    assert len(calls) == 1
    assert retriever.cache_hits == 1 and retriever.cache_misses == 1


def check_transient_retrieval_retry() -> None:
    retriever = AcademicRetriever(
        RetrievalConfig(
            use_openalex=False,
            use_semantic_scholar=False,
            use_arxiv=False,
            use_serper=False,
            pasa_id2paper_path="",
        )
    )
    calls = 0

    def flaky_search(query: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.ConnectionError("temporary reset")
        return [Paper(paper_id="recovered", title="Recovered paper")]

    papers = retriever._safe(flaky_search, "retry query")
    assert calls == 2
    assert papers and papers[0].paper_id == "recovered"
    assert not retriever.warnings


def check_selector_training_parser() -> None:
    record = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Searched Paper:\nTitle: Relevant Work\nAbstract: A useful abstract.\n\n"
                    "User Query: Find useful work.\n\nOutput format: Decision: True/False"
                ),
            },
            {"role": "assistant", "content": "True\nReason: direct match"},
        ]
    }
    example = parse_selector_record(record)
    assert example.query == "Find useful work."
    assert example.document.startswith("Relevant Work\n")
    assert example.label == 1


def check_dense_title_benchmark_helpers() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.00001v2") == "2401.00001"
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "d", "a"]],
        weights=[1.0, 1.0],
        rrf_k=60,
    )
    assert fused[:2] == ["b", "a"]
    examples = [{"gold_ids": ["a", "d"]}]
    metrics = recall_metrics(examples, [fused], [2, 4])
    assert metrics["macro_recall@2"] == 0.5
    assert metrics["micro_recall@4"] == 1.0
    gate = quality_gate(
        {
            "lexical": {"macro_recall@20": 0.10, "macro_recall@100": 0.15},
            "rrf_fusion": {"macro_recall@20": 0.105, "macro_recall@100": 0.18},
        }
    )
    assert gate["accepted_for_end_to_end_trial"]


def check_title_rrf_fusion() -> None:
    lexical = [
        Paper(paper_id="a", title="A", api_score=0.9),
        Paper(paper_id="b", title="B", api_score=0.8),
    ]
    dense = [
        Paper(paper_id="b", title="B", api_score=0.7),
        Paper(paper_id="c", title="C", api_score=0.6),
    ]
    fused = fuse_title_results([lexical, dense], limit=3, rrf_k=60)
    assert [paper.paper_id for paper in fused] == ["b", "a", "c"]
    assert fused[0].api_score == 1.0


def check_dense_head_protection() -> None:
    baseline = [
        Paper(paper_id=f"api-{index}", title=f"API paper {index}", source="OpenAlex")
        for index in range(25)
    ]
    dense = [
        Paper(paper_id=f"dense-{index}", title=f"Dense paper {index}", source="DenseTitleDB")
        for index in range(10)
    ]
    merged = merge_baseline_head(baseline, dense + baseline, 20)
    assert [paper.paper_id for paper in merged[:20]] == [
        paper.paper_id for paper in baseline[:20]
    ]
    assert any(is_dense_only_candidate(paper) for paper in merged[20:])
    assert is_dense_only_candidate(dense[0]) is True
    assert is_dense_only_candidate(
        Paper(paper_id="cross", title="Cross", source="DenseTitleDB+OpenAlex")
    ) is False

    short = merge_baseline_head(baseline[:1], dense[:2] + baseline[:1], 2)
    assert len(short) == 3
    assert short[0].paper_id == "api-0"


def check_asta_paper_finder_adapter() -> None:
    fixture = ROOT / "runs" / "asta_adapter_fixture.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        json.dumps(
            [
                {
                    "input": {"query_id": "specific_demo", "query": "find the target paper"},
                    "scorer_criteria": {"corpus_ids": ["12345"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    try:
        samples = load_asta_samples(fixture)
    finally:
        fixture.unlink(missing_ok=True)
    assert len(samples) == 1 and samples[0].query_id == "specific_demo"

    direct = Paper(
        paper_id="a" * 40,
        title="Target Paper",
        abstract="Verbatim abstract evidence.",
        year=2024,
        source_ids=["CorpusId:12345"],
    )
    doi_paper = Paper(paper_id="openalex:1", title="DOI Paper", doi="10.1000/example")
    assert direct_corpus_id(direct) == "12345"
    assert semantic_scholar_lookup_id(doi_paper) == "DOI:10.1000/example"

    output = build_asta_output("specific_demo", [(direct, "12345")], limit=30)
    result_rows = output["output"]["results"]
    assert result_rows[0]["paper_id"] == "12345"
    assert "Title: Target Paper" in result_rows[0]["markdown_evidence"]
    assert "Verbatim abstract evidence." in result_rows[0]["markdown_evidence"]

    sample = AstaSample(
        query_id="specific_demo",
        query="find the target paper",
        scorer_criteria={"corpus_ids": ["12345", "67890"]},
        raw={},
    )
    metrics = diagnostic_metrics(
        sample,
        result_rows,
        retrieved_count=2,
        agent_api_calls=3,
        llm_calls=1,
        resolver_api_calls=1,
        resolver_failures=0,
        latency_seconds=2.0,
    )
    assert extract_known_corpus_ids(sample.scorer_criteria) == {"12345", "67890"}
    assert metrics["known_hits@30"] == 1.0
    assert metrics["known_coverage@30"] == 0.5


def check_exact_bibliographic_routing() -> None:
    assert _looks_like_bibliographic_lookup("BART by Lewis et al.")
    assert _looks_like_bibliographic_lookup("the MS^2 DeYong2021 paper")
    assert _looks_like_bibliographic_lookup("the Multi-news fabbri2019multinews paper")
    assert not _looks_like_bibliographic_lookup(
        "Which papers investigate clustering-based efficient attention in transformers?"
    )
    assert not RetrievalConfig().use_semantic_scholar_exact_match


def check_author_metadata_routing() -> None:
    assert _looks_like_author_metadata_lookup("a Nature portfolio papers by David Harel")
    assert _looks_like_author_metadata_lookup(
        'NAACL 2010 or 2012 papers co-authored by one of the authors of the "BERT" paper'
    )
    assert not _looks_like_author_metadata_lookup("BART by Lewis et al.")
    names = _author_names_from_queries([
        "NAACL 2010 papers by Jacob Devlin",
        "NAACL 2012 papers by Kristina Toutanova",
        "NAACL 2010 papers by Jacob Devlin",
    ])
    assert names == ["Jacob Devlin", "Kristina Toutanova"]
    assert not RetrievalConfig().use_openalex_metadata_constraints


def check_metadata_source_hygiene_routing() -> None:
    metadata_plan = QueryPlan(
        original_query="Claire Cardie ACL papers published at 2014 or 2017",
        constraints={"venue": "ACL", "year": [2014, 2017], "author": "Claire Cardie"},
    )
    assert _requires_verifiable_metadata(metadata_plan.original_query, metadata_plan)
    citation_plan = QueryPlan(
        original_query="Papers citing DistilBERT after 2022 with more than 50 citations",
        constraints={"cites": "DistilBERT", "min_citations": 50},
    )
    assert _requires_verifiable_metadata(citation_plan.original_query, citation_plan)
    topical_plan = QueryPlan(
        original_query="retrieval augmented generation architectures",
        constraints={"topic": "RAG"},
    )
    assert not _requires_verifiable_metadata(topical_plan.original_query, topical_plan)
    exact_plan = QueryPlan(original_query="BART by Lewis et al.")
    assert not _requires_verifiable_metadata(exact_plan.original_query, exact_plan)
    assert RetrievalConfig().suppress_title_only_for_metadata_queries is False
    assert RetrievalConfig().metadata_direct_shortcut is False


def check_asta_official_output_policy() -> None:
    exact = Paper(paper_id="exact", title="Exact", source="SemanticScholarExact+OpenAlex")
    topical = Paper(paper_id="topical", title="Topical", source="OpenAlex")
    noisy = [(topical, "2"), (exact, "1")]
    assert official_resolved_results(noisy, 100) == [(exact, "1")]
    assert official_resolved_results([(topical, "2")], 100) == [(topical, "2")]
    assert official_resolved_results([], 100) == []


def check_smoke_command() -> None:
    out = ROOT / "runs" / "offline_quality_smoke.json"
    cmd = [
        sys.executable,
        "run_agent.py",
        "--config",
        "config.smoke.json",
        "--query",
        "large language model hallucination detection factuality evaluation",
        "--output",
        str(out),
        "--no_llm",
        "--fallback_models",
        "--top_k",
        "5",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["papers"], "smoke query should return sample papers"
    assert data["papers"][0]["paper_id"] == "P1"
    assert data["agent_trace"], "agent trace should document crawler/selector/ranker steps"
    assert {"authority_score", "recency_score", "diversity_score"} <= set(data["papers"][0])
    assert {"cache_hits", "cache_misses", "retrieval_rounds", "stopped_early", "stop_reason"} <= set(data["stats"])


if __name__ == "__main__":
    main()
