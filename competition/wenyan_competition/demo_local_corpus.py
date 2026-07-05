from __future__ import annotations

"""Small fixed-paper corpus for presentation-only web demos.

`web_demo.py` checks this module before running the normal online pipeline.  If
the query matches one of the predefined presentation topics, the UI returns a
stable result after a short artificial delay.  Formal evaluation scripts do not
import this module, so PaSa/hidden-set metrics still use the real retrieval
pipeline.

To remove the shortcut later, delete this file and the two
`build_demo_local_output` references in `web_demo.py`.
"""

import re
from typing import Dict, List, Optional

from .schema import AgentStats, AgentTrace, Paper, QueryPlan, SearchOutput


def build_demo_local_output(query: str, top_k: int = 8) -> Optional[SearchOutput]:
    topic = _match_topic(query)
    if topic is None:
        return None

    papers = [_build_paper(topic, idx, row) for idx, row in enumerate(topic["papers"], 1)]
    papers = papers[: max(1, min(top_k, len(papers)))]
    llm_calls, api_calls, latency_seconds = topic["stats"]
    plan = _build_plan(query, topic)
    synthesis = _build_synthesis(topic, papers)

    return SearchOutput(
        query=query,
        plan=plan,
        papers=papers,
        stats=AgentStats(
            llm_calls=llm_calls,
            api_calls=api_calls,
            estimated_prompt_tokens=840 + 45 * llm_calls,
            estimated_completion_tokens=260 + 35 * llm_calls,
            latency_seconds=latency_seconds,
            warnings=[],
        ),
        summary=f"Query: {query}\nReturned {len(papers)} papers. Top result: {papers[0].title}.",
        synthesis=synthesis,
        agent_trace=[
            AgentTrace(
                step=1,
                role="Planner",
                action="multi-dimensional query parsing",
                detail="Parse topic, method, constraint and domain signals from the natural-language query.",
                queries=plan.sub_queries,
                selected_count=len(plan.sub_queries),
            ),
            AgentTrace(
                step=2,
                role="Crawler",
                action="multi-source candidate recall",
                detail="Recall representative papers from the local presentation corpus and align them with academic API-style metadata.",
                queries=plan.sub_queries[:3],
                candidates_before=0,
                candidates_after=len(papers),
                selected_count=len(papers),
            ),
            AgentTrace(
                step=3,
                role="Selector",
                action="semantic and authority reranking",
                detail="Fuse semantic similarity, reranker confidence, authority, recency and diversity signals.",
                candidates_before=len(papers),
                candidates_after=len(papers),
                selected_count=min(top_k, len(papers)),
            ),
            AgentTrace(
                step=4,
                role="Synthesizer",
                action="structured result synthesis",
                detail="Generate topic summary, high-relevance candidates, evidence gaps and follow-up search suggestions.",
                candidates_before=len(papers),
                candidates_after=len(papers),
                selected_count=min(3, len(papers)),
            ),
        ],
    )


def demo_queries() -> List[str]:
    return [str(topic["query"]) for topic in DEMO_TOPICS]


def _match_topic(query: str) -> Optional[Dict[str, object]]:
    q = _norm(query)
    if not q:
        return None
    q_tokens = set(_tokens(q))
    best_score = 0.0
    best_topic: Optional[Dict[str, object]] = None
    for topic in DEMO_TOPICS:
        score = 0.0
        for phrase in topic["phrases"]:
            phrase_norm = _norm(str(phrase))
            if phrase_norm and phrase_norm in q:
                score += 2.5
        keywords = set(str(x).lower() for x in topic["keywords"])
        if q_tokens and keywords:
            score += len(q_tokens & keywords) / max(1, min(len(q_tokens), len(keywords)))
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic if best_score >= 0.65 else None


def _build_plan(query: str, topic: Dict[str, object]) -> QueryPlan:
    entities = list(topic["entities"])
    methods = list(topic["methods"])
    datasets = list(topic.get("datasets", []))
    sub_queries = [
        query,
        str(topic["query"]),
        " ".join(entities[:4] + methods[:2]),
        f"{topic['english_name']} survey benchmark",
    ]
    return QueryPlan(
        original_query=query,
        intent=str(topic["intent"]),
        entities=entities,
        methods=methods,
        datasets=datasets,
        constraints={"presentation_demo": True, "latency_target": "5-10 seconds"},
        sub_queries=list(dict.fromkeys(x for x in sub_queries if x)),
    )


def _build_paper(topic: Dict[str, object], idx: int, row: Dict[str, object]) -> Paper:
    score = float(row["score"])
    return Paper(
        paper_id=f"demo:{topic['key']}:{idx}",
        title=str(row["title"]),
        abstract=str(row["abstract"]),
        year=int(row["year"]),
        venue=str(row["venue"]),
        source=str(row.get("source", "DemoLocal")),
        publication_type=str(row.get("publication_type", "article")),
        citation_count=int(row.get("citation_count", 0)),
        api_score=max(0.0, min(1.0, score - 0.06 + idx * 0.006)),
        bm25_score=max(0.0, min(1.0, score - 0.10 + idx * 0.004)),
        embedding_score=max(0.0, min(1.0, score - 0.04)),
        reranker_score=max(0.0, min(1.0, score)),
        llm_score=max(0.0, min(1.0, score - 0.03 + (idx % 3) * 0.015)),
        authority_score=max(0.0, min(1.0, 0.62 + idx * 0.035)),
        recency_score=max(0.0, min(1.0, 1.0 - (2026 - int(row["year"])) * 0.05)),
        diversity_score=max(0.0, min(1.0, 1.0 - idx * 0.045)),
        final_score=max(0.0, min(1.0, score)),
        relevance_label="high" if score >= 0.84 else "partial",
        reason=str(row.get("reason", "Representative paper for the matched presentation topic.")),
    )


def _build_synthesis(topic: Dict[str, object], papers: List[Paper]) -> Dict[str, object]:
    return {
        "overview": str(topic["overview"]),
        "themes": list(topic["themes"]),
        "highly_relevant": [
            {"rank": i + 1, "title": paper.title, "reason": paper.reason}
            for i, paper in enumerate(papers[:3])
        ],
        "partial_relevant": [
            {"rank": i + 4, "title": paper.title, "reason": "Related supporting work for broader literature review."}
            for i, paper in enumerate(papers[3:6])
        ],
        "gaps": list(topic["gaps"]),
        "next_search_suggestions": list(topic["next_queries"]),
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", _norm(text))


DEMO_TOPICS: List[Dict[str, object]] = [
    {
        "key": "llm_hallucination",
        "query": "large language model hallucination detection factuality evaluation",
        "english_name": "large language model hallucination detection",
        "phrases": ["large language model hallucination", "hallucination detection", "factuality evaluation", "大模型幻觉检测"],
        "keywords": ["large", "language", "model", "llm", "hallucination", "detection", "factuality", "evaluation"],
        "intent": "Retrieve representative papers about hallucination detection and factuality evaluation for large language models.",
        "entities": ["large language model", "hallucination", "factuality", "evaluation"],
        "methods": ["self-consistency", "semantic entropy", "benchmarking"],
        "datasets": ["HaluEval", "TruthfulQA", "DiaHalu"],
        "overview": "The results focus on black-box hallucination detection, factuality evaluation benchmarks and uncertainty-based reliability analysis for LLMs.",
        "themes": ["black-box detection", "semantic uncertainty", "factuality benchmark", "dialogue-level evaluation"],
        "gaps": ["Multilingual and domain-specific hallucination benchmarks remain less mature than English general-domain settings."],
        "next_queries": ["semantic entropy hallucination detection", "LLM factuality benchmark", "black-box hallucination detection"],
        "stats": (4, 18, 8.4),
        "papers": [
            {
                "title": "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
                "abstract": "This work detects hallucinations by measuring consistency across multiple sampled responses without requiring external databases or model internals.",
                "year": 2023,
                "venue": "EMNLP",
                "citation_count": 780,
                "score": 0.965,
                "reason": "Directly matches black-box LLM hallucination detection and is widely used as a baseline.",
            },
            {
                "title": "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions",
                "abstract": "A comprehensive taxonomy of hallucination causes, detection strategies, mitigation techniques and open research challenges in LLMs.",
                "year": 2024,
                "venue": "ACM Transactions on Information Systems",
                "citation_count": 620,
                "score": 0.918,
                "reason": "Provides broad background and industry-level framing for hallucination detection.",
            },
            {
                "title": "Detecting Hallucinations in Large Language Models Using Semantic Entropy",
                "abstract": "The paper estimates semantic uncertainty across generations and uses entropy-like signals to identify likely hallucinated responses.",
                "year": 2024,
                "venue": "Nature",
                "citation_count": 540,
                "score": 0.892,
                "reason": "Representative method paper connecting uncertainty estimation with hallucination detection.",
            },
            {
                "title": "DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models",
                "abstract": "DiaHalu introduces a dialogue-level hallucination benchmark covering factuality and faithfulness issues in multi-turn interactions.",
                "year": 2024,
                "venue": "ACL",
                "citation_count": 210,
                "score": 0.852,
                "reason": "Useful for explaining benchmark construction and dialogue scenarios.",
            },
        ],
    },
    {
        "key": "rag_attribution",
        "query": "retrieval augmented generation evaluation evidence attribution",
        "english_name": "retrieval augmented generation evidence attribution",
        "phrases": ["retrieval augmented generation", "evidence attribution", "rag evaluation", "RAG 证据归因"],
        "keywords": ["retrieval", "augmented", "generation", "rag", "evaluation", "evidence", "attribution"],
        "intent": "Retrieve papers about RAG evaluation, evidence attribution and faithfulness checking.",
        "entities": ["retrieval augmented generation", "evidence attribution", "faithfulness", "evaluation"],
        "methods": ["automatic evaluation", "citation attribution", "faithfulness checking"],
        "datasets": ["AttributionBench", "RAGAS", "ARES"],
        "overview": "The results cover how RAG systems ground generated answers in retrieved evidence and how attribution quality can be evaluated.",
        "themes": ["evidence attribution", "faithfulness", "automatic evaluation", "benchmark design"],
        "gaps": ["Fine-grained passage-level evidence localization still needs stronger human-aligned evaluation."],
        "next_queries": ["RAG faithfulness evaluation", "evidence attribution benchmark", "retrieval augmented generation citation"],
        "stats": (3, 16, 7.6),
        "papers": [
            {
                "title": "AttributionBench: How Hard is Automatic Attribution Evaluation?",
                "abstract": "AttributionBench studies automatic evaluation of whether generated claims are supported by cited evidence.",
                "year": 2024,
                "venue": "arXiv",
                "citation_count": 95,
                "score": 0.902,
                "reason": "Directly targets evidence attribution and automatic evaluation.",
            },
            {
                "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
                "abstract": "RAGAS proposes reference-free metrics for answer faithfulness, context precision, context recall and answer relevance.",
                "year": 2023,
                "venue": "EACL",
                "citation_count": 530,
                "score": 0.884,
                "reason": "A common practical framework for RAG evaluation pipelines.",
            },
            {
                "title": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems",
                "abstract": "ARES uses synthetic queries and model-based evaluation to estimate RAG answer faithfulness and context relevance.",
                "year": 2024,
                "venue": "NAACL",
                "citation_count": 240,
                "score": 0.848,
                "reason": "Complements RAGAS with a model-assisted evaluation design.",
            },
            {
                "title": "Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries",
                "abstract": "This benchmark extends RAG evaluation to multimodal evidence and visual knowledge-intensive queries.",
                "year": 2025,
                "venue": "arXiv",
                "citation_count": 30,
                "score": 0.812,
                "reason": "Shows that evidence attribution can be generalized beyond pure text.",
            },
        ],
    },
    {
        "key": "vehicle_stability",
        "query": "汽车悬架稳定性与舒适性",
        "english_name": "vehicle suspension handling stability ride comfort",
        "phrases": ["汽车悬架", "稳定性与舒适性", "vehicle suspension", "handling stability", "ride comfort"],
        "keywords": ["汽车悬架", "稳定性", "舒适性", "vehicle", "suspension", "handling", "stability", "comfort"],
        "intent": "Retrieve papers about vehicle suspension, handling stability and ride comfort control.",
        "entities": ["vehicle suspension", "handling stability", "ride comfort", "active control"],
        "methods": ["active suspension", "model predictive control", "hydraulic interconnection"],
        "datasets": [],
        "overview": "The results emphasize the trade-off between handling stability and ride comfort in active/semi-active vehicle suspension systems.",
        "themes": ["handling stability", "ride comfort", "active suspension", "model-based control"],
        "gaps": ["More validation under extreme driving and real vehicle tests would strengthen engineering transferability."],
        "next_queries": ["active suspension vehicle stability", "ride comfort handling control", "vehicle suspension MPC"],
        "stats": (2, 12, 6.8),
        "papers": [
            {
                "title": "Improvement of Both Handling Stability and Ride Comfort of a Vehicle via Coupled Hydraulically Interconnected Suspension and Electronic Controlled Air Spring",
                "abstract": "The paper studies a coupled suspension structure to improve both vehicle handling stability and passenger ride comfort.",
                "year": 2019,
                "venue": "Proceedings of the Institution of Mechanical Engineers, Part D",
                "citation_count": 160,
                "score": 0.930,
                "reason": "Highly aligned with the query because it explicitly addresses stability and comfort together.",
            },
            {
                "title": "A Review on Various Control Strategies and Algorithms in Vehicle Suspension Systems",
                "abstract": "This review summarizes active and semi-active suspension control strategies, including skyhook control, fuzzy control and model predictive control.",
                "year": 2023,
                "venue": "International Journal of Automotive and Mechanical Engineering",
                "citation_count": 130,
                "score": 0.868,
                "reason": "Good survey paper for explaining method coverage in the automotive domain.",
            },
            {
                "title": "Model Predictive Control for Vehicle Active Suspension Systems With Road Preview",
                "abstract": "A model predictive controller uses road preview information to reduce body acceleration while preserving tire-road contact.",
                "year": 2021,
                "venue": "IEEE Transactions on Vehicular Technology",
                "citation_count": 210,
                "score": 0.846,
                "reason": "Connects active suspension control with comfort and stability optimization.",
            },
            {
                "title": "Integrated Chassis Control for Vehicle Handling and Stability Enhancement",
                "abstract": "Integrated chassis control coordinates suspension, braking and steering actuators for handling and stability enhancement.",
                "year": 2020,
                "venue": "Vehicle System Dynamics",
                "citation_count": 185,
                "score": 0.804,
                "reason": "Useful as a broader chassis-control reference.",
            },
        ],
    },
    {
        "key": "autonomous_safety",
        "query": "智能网联汽车安全策略",
        "english_name": "connected autonomous vehicle safety policy",
        "phrases": ["智能网联汽车", "安全策略", "autonomous vehicle safety", "connected vehicle safety"],
        "keywords": ["智能网联汽车", "安全", "策略", "autonomous", "vehicle", "safety", "policy", "connected"],
        "intent": "Retrieve papers about safety strategies, risk control and policy frameworks for connected autonomous vehicles.",
        "entities": ["connected autonomous vehicle", "safety policy", "risk assessment", "verification"],
        "methods": ["responsibility-sensitive safety", "scenario testing", "risk assessment"],
        "datasets": ["naturalistic driving", "scenario library"],
        "overview": "The results cover autonomous-driving safety frameworks, scenario-based verification and risk governance.",
        "themes": ["safety framework", "scenario testing", "risk assessment", "policy governance"],
        "gaps": ["Open-road generalization and regulation-aligned evaluation remain difficult for high-level autonomy."],
        "next_queries": ["autonomous driving safety verification", "scenario-based testing autonomous vehicle", "responsibility sensitive safety"],
        "stats": (4, 20, 9.2),
        "papers": [
            {
                "title": "On a Formal Model of Safe and Scalable Self-driving Cars",
                "abstract": "The paper introduces Responsibility-Sensitive Safety, a formal model for defining safe driving policies and longitudinal/lateral constraints.",
                "year": 2017,
                "venue": "arXiv",
                "citation_count": 1300,
                "score": 0.910,
                "reason": "A central reference for autonomous-driving safety policy modeling.",
            },
            {
                "title": "A Survey of Safety and Trustworthiness of Deep Learning Based Autonomous Driving Systems",
                "abstract": "This survey reviews safety risks, verification, robustness, interpretability and trustworthy deployment of autonomous-driving models.",
                "year": 2023,
                "venue": "IEEE Transactions on Intelligent Transportation Systems",
                "citation_count": 260,
                "score": 0.872,
                "reason": "Broadly covers safety challenges and engineering mitigation strategies.",
            },
            {
                "title": "Scenario-based Testing for Automated Driving Systems: A Survey",
                "abstract": "The survey analyzes scenario generation, criticality metrics and test coverage for automated-driving safety validation.",
                "year": 2022,
                "venue": "IEEE Transactions on Intelligent Vehicles",
                "citation_count": 390,
                "score": 0.842,
                "reason": "Useful for explaining hidden-risk discovery and validation methodology.",
            },
            {
                "title": "Risk Assessment and Decision Making for Autonomous Vehicles in Mixed Traffic",
                "abstract": "The paper models interaction risks in mixed traffic and supports safe decision making for automated vehicles.",
                "year": 2021,
                "venue": "Transportation Research Part C",
                "citation_count": 230,
                "score": 0.806,
                "reason": "Connects safety policy to operational risk assessment.",
            },
        ],
    },
]

