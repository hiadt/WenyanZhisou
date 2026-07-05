from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from wenyan_competition.config import RetrievalConfig, load_config
from wenyan_competition.dataset import extract_gold_items
from wenyan_competition.llm import _as_dict, _as_list, heuristic_plan, heuristic_synthesis
from wenyan_competition.retrievers import (
    AcademicRetriever,
    _arxiv_id_from_url,
    _arxiv_queries,
    _extract_arxiv_ids_from_serper,
    _openalex_api_work_url,
    _serper_arxiv_queries,
)
from wenyan_competition.schema import Paper
from evaluate_pasa import _apply_formal_eval_defaults, flexible_recall_at, paper_aliases


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
        check_local_search_can_skip_online_sources,
        check_arxiv_query_helpers,
        check_serper_arxiv_helpers,
        check_formal_eval_defaults,
        check_openalex_url_normalization,
        check_citation_fetch_warnings_are_quiet,
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
    assert flexible_recall_at([paper_aliases(by_arxiv)], gold_items, 1) == 1.0
    assert flexible_recall_at([paper_aliases(by_title)], gold_items, 1) == 1.0


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


def check_local_search_can_skip_online_sources() -> None:
    tmp = ROOT / "runs" / "tiny_id2paper.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(
        json.dumps({"2309.04564": "When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale"}),
        encoding="utf-8",
    )
    retriever = AcademicRetriever(
        RetrievalConfig(
            use_openalex=True,
            use_semantic_scholar=True,
            use_arxiv=True,
            use_serper=False,
            pasa_id2paper_path=str(tmp),
            pasa_title_limit=10,
        )
    )
    papers = retriever.search_many(["data pruning pretraining LLM"], include_online=False)
    assert any(p.paper_id == "2309.04564" for p in papers)
    assert retriever.api_calls == 0


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
    assert cfg.ranking.llm_verify_top_n == 60
    assert cfg.ranking.llm_verifier_batch_size >= 60
    assert cfg.budget.max_llm_calls_per_query == 2
    assert cfg.retrieval.max_candidates == 220
    assert cfg.retrieval.pasa_title_limit >= 120
    assert cfg.retrieval.max_rounds == 1
    assert cfg.retrieval.citation_expand_limit == 0
    assert cfg.budget.max_api_calls_per_query == 36
    assert cfg.retrieval.serper_query_limit <= 2
    assert cfg.retrieval.serper_query_variants <= 2
    assert cfg.retrieval.arxiv_query_limit <= 2
    assert cfg.retrieval.arxiv_query_variants <= 2


def check_openalex_url_normalization() -> None:
    assert _openalex_api_work_url("https://openalex.org/W123") == "https://api.openalex.org/works/W123"
    assert _openalex_api_work_url("W456") == "https://api.openalex.org/works/W456"


def check_citation_fetch_warnings_are_quiet() -> None:
    text = (ROOT / "wenyan_competition" / "retrievers.py").read_text(encoding="utf-8")
    assert "fetch_openalex_work, paper_id, warn=False" in text
    assert "fetch_semantic_scholar_paper, paper_id, warn=False" in text


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


if __name__ == "__main__":
    main()
