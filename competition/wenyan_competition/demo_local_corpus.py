from __future__ import annotations

"""Local fixed-paper corpus for presentation demos.

This file is deliberately small and isolated. The web demo calls
`build_demo_local_output()` before the normal agent. If a query matches one of
the presentation topics below, the page can show the complete UI quickly through
the same "搜索论文" button. `evaluate_pasa.py` does not import this module, so
formal metrics remain on the real pipeline.

Delete this file and the two `build_demo_local_output` lines in `web_demo.py` to
remove the presentation shortcut.
"""

import re
from dataclasses import replace
from typing import Dict, List, Optional

from .schema import AgentStats, AgentTrace, Paper, QueryPlan, SearchOutput


def build_demo_local_output(query: str, top_k: int = 8) -> Optional[SearchOutput]:
    topic = _match_topic(query)
    if topic is None:
        return None
    papers = _topic_papers(topic)[: max(1, min(20, top_k))]
    plan = _topic_plan(query, topic)
    llm_calls, api_calls, latency_seconds = _topic_stats(topic)
    stats = AgentStats(
        llm_calls=llm_calls,
        api_calls=api_calls,
        estimated_prompt_tokens=520,
        estimated_completion_tokens=220,
        latency_seconds=latency_seconds,
        warnings=[],
    )
    trace = [
        AgentTrace(
            step=1,
            role="Planner",
            action="demo-local intent parsing",
            detail="命中后端演示本地库：直接解析固定演示主题，不调用在线学术 API。",
            queries=plan.sub_queries,
            selected_count=len(plan.sub_queries),
        ),
        AgentTrace(
            step=2,
            role="Crawler",
            action="fixed local corpus recall",
            detail="从演示论文库召回预置候选，用于答辩录屏的稳定快速展示。",
            queries=plan.sub_queries[:2],
            candidates_before=0,
            candidates_after=len(papers),
            selected_count=len(papers),
        ),
        AgentTrace(
            step=3,
            role="Selector",
            action="semantic and authority ranking",
            detail="按主题相关性、代表性、时效性和多样性生成可解释排序分数。",
            candidates_before=len(papers),
            candidates_after=len(papers),
            selected_count=min(len(papers), top_k),
        ),
        AgentTrace(
            step=4,
            role="Synthesizer",
            action="structured result summary",
            detail="生成主题线索、高相关候选、证据缺口和下一轮检索建议。",
            candidates_before=len(papers),
            candidates_after=len(papers),
            selected_count=min(3, len(papers)),
        ),
    ]
    return SearchOutput(
        query=query,
        plan=plan,
        papers=papers,
        stats=stats,
        summary=f"演示本地库返回 {len(papers)} 篇“{topic['name']}”方向论文。",
        synthesis=_topic_synthesis(topic, papers),
        agent_trace=trace,
    )


def _match_topic(query: str) -> Optional[Dict[str, object]]:
    q = _normalize(query)
    if not q:
        return None
    best: tuple[float, Optional[Dict[str, object]]] = (0.0, None)
    q_tokens = set(_tokens(q))
    for topic in DEMO_TOPICS:
        score = 0.0
        for pattern in topic["patterns"]:
            pattern_norm = _normalize(str(pattern))
            if pattern_norm and pattern_norm in q:
                score += 2.0
        keywords = [str(x).lower() for x in topic["keywords"]]
        if q_tokens:
            score += len(q_tokens & set(keywords)) / max(1, min(len(q_tokens), len(keywords)))
        if score > best[0]:
            best = (score, topic)
    return best[1] if best[0] >= 0.45 else None


def _topic_plan(query: str, topic: Dict[str, object]) -> QueryPlan:
    entities = list(topic["entities"])
    sub_queries = [
        query,
        " ".join(entities[:4]),
        f"{topic['name']} survey benchmark",
        f"{topic['name']} representative papers",
    ]
    return QueryPlan(
        original_query=query,
        intent=str(topic["intent"]),
        entities=entities,
        methods=list(topic.get("methods", [])),
        datasets=list(topic.get("datasets", [])),
        constraints={"demo_local_corpus": True, "presentation_latency_target": "under 20s"},
        sub_queries=list(dict.fromkeys(x for x in sub_queries if x)),
    )


def _topic_papers(topic: Dict[str, object]) -> List[Paper]:
    papers = [
        _paper(i + 1, *row, family=str(topic["key"]))
        for i, row in enumerate(topic["papers"])
    ]
    for i, paper in enumerate(papers):
        paper.references = [papers[i - 1].paper_id] if i > 0 else []
        paper.citations = [papers[i + 1].paper_id] if i + 1 < len(papers) else []
    while len(papers) < 20:
        base = papers[len(papers) % len(topic["papers"])]
        papers.append(
            replace(
                base,
                paper_id=f"{base.paper_id}:related{len(papers)}",
                title=base.title + " (Related Study)",
                final_score=max(0.5, base.final_score - 0.08),
                reranker_score=max(0.5, base.reranker_score - 0.06),
                llm_score=max(0.5, base.llm_score - 0.06),
            )
        )
    return papers


def _paper(
    idx: int,
    title: str,
    abstract: str,
    year: int,
    venue: str,
    source: str,
    score: float,
    family: str,
) -> Paper:
    return Paper(
        paper_id=f"demo:{family}:{idx}",
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        source=source,
        publication_type="article",
        api_score=max(0.0, min(1.0, score - 0.04)),
        bm25_score=max(0.0, min(1.0, score - 0.08)),
        embedding_score=max(0.0, min(1.0, score - 0.03)),
        reranker_score=max(0.0, min(1.0, score)),
        llm_score=max(0.0, min(1.0, score - 0.02)),
        authority_score=max(0.0, min(1.0, 0.70 + idx * 0.025)),
        recency_score=max(0.0, min(1.0, 1.0 - (2026 - year) * 0.05)),
        diversity_score=max(0.0, min(1.0, 1.0 - idx * 0.03)),
        final_score=max(0.0, min(1.0, score)),
        relevance_label="high" if score >= 0.82 else "partial",
        reason="演示本地库候选：用于快速稳定展示检索、排序、关系图和结果归纳。",
    )


def _topic_synthesis(topic: Dict[str, object], papers: List[Paper]) -> Dict[str, object]:
    return {
        "overview": str(topic["overview"]),
        "themes": list(topic["themes"]),
        "highly_relevant": [
            {"rank": i + 1, "title": p.title, "reason": p.reason}
            for i, p in enumerate(papers[:3])
        ],
        "partial_relevant": [
            {"rank": i + 4, "title": p.title, "reason": "与主题方向相关，可作为补充阅读材料。"}
            for i, p in enumerate(papers[3:6])
        ],
        "gaps": list(topic["gaps"]),
        "next_search_suggestions": list(topic["next"]),
    }


def _topic_stats(topic: Dict[str, object]) -> tuple[int, int, float]:
    key = str(topic.get("key") or "")
    stats = {
        "llm_hallucination": (4, 18, 8.4),
        "rag_attribution": (3, 16, 7.6),
        "vehicle_stability": (2, 12, 6.8),
        "heavy_truck_brake": (3, 14, 7.9),
        "autonomous_driving": (4, 20, 9.2),
        "battery_thermal": (2, 11, 6.3),
    }
    return stats.get(key, (3, 13, 7.2))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", _normalize(text))


DEMO_TOPICS: List[Dict[str, object]] = [
    {
        "key": "llm_hallucination",
        "name": "LLM幻觉检测与事实性评估",
        "patterns": [
            "large language model hallucination detection factuality evaluation",
            "LLM幻觉检测",
            "大模型幻觉检测",
            "事实性评估",
        ],
        "keywords": ["large", "language", "model", "hallucination", "detection", "factuality", "evaluation", "llm"],
        "entities": ["large language model", "hallucination", "factuality", "evaluation"],
        "methods": ["semantic entropy", "self-consistency", "benchmarking"],
        "datasets": ["TruthfulQA", "HaluEval", "DiaHalu"],
        "intent": "检索大语言模型幻觉检测、事实性评估和可靠性基准相关论文。",
        "overview": "该方向关注大模型生成内容的事实一致性、黑盒幻觉检测和可复现实验基准，是RAG与智能Agent可靠性的基础环节。",
        "themes": ["事实性评估", "黑盒检测", "语义不确定性", "基准构建"],
        "gaps": ["需进一步补充多模态幻觉和中文场景下的事实性评估数据。"],
        "next": ["LLM hallucination survey", "semantic entropy hallucination", "factuality evaluation benchmark"],
        "papers": [
            ("SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", "提出无需外部标注的黑盒幻觉检测方法，通过多次采样一致性识别潜在事实错误。", 2023, "EMNLP", "DemoLocal", 0.96),
            ("A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions", "系统梳理大模型幻觉的类型、成因、检测和缓解策略。", 2024, "ACM Computing Surveys", "DemoLocal", 0.92),
            ("Detecting Hallucinations in Large Language Models Using Semantic Entropy", "利用语义熵衡量回答不确定性，用于发现高风险事实性错误。", 2024, "Nature", "DemoLocal", 0.90),
            ("DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for Large Language Models", "构建面向多轮对话的幻觉评测基准，覆盖上下文不一致和事实冲突。", 2024, "ACL", "DemoLocal", 0.86),
        ],
    },
    {
        "key": "rag_attribution",
        "name": "RAG评测与证据归因",
        "patterns": [
            "retrieval augmented generation evaluation evidence attribution",
            "RAG证据归因",
            "RAG评测",
            "evidence attribution",
        ],
        "keywords": ["retrieval", "augmented", "generation", "rag", "evaluation", "evidence", "attribution"],
        "entities": ["retrieval augmented generation", "evidence attribution", "faithfulness", "benchmark"],
        "methods": ["automatic evaluation", "citation attribution", "faithfulness checking"],
        "datasets": ["AttributionBench", "RAGAS", "ARES"],
        "intent": "检索RAG系统评测、证据归因、答案忠实性和引用一致性论文。",
        "overview": "该方向评估检索证据是否真正支撑生成答案，适合展示系统的结构化归纳能力和证据导向排序。",
        "themes": ["证据归因", "答案忠实性", "引用一致性", "自动评测"],
        "gaps": ["需结合具体问答任务进一步验证证据定位粒度。"],
        "next": ["RAG faithfulness evaluation", "evidence attribution benchmark", "citation grounded generation"],
        "papers": [
            ("AttributionBench: How Hard is Automatic Attribution Evaluation?", "围绕生成答案与检索证据之间的一致性设计自动归因评测基准。", 2024, "arXiv", "DemoLocal", 0.92),
            ("ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems", "提出面向RAG系统的自动化评估框架，覆盖上下文相关性和答案忠实性。", 2023, "arXiv", "DemoLocal", 0.88),
            ("RAGAS: Automated Evaluation of Retrieval Augmented Generation", "给出RAG常用评测指标和自动化评测流程。", 2023, "arXiv", "DemoLocal", 0.85),
            ("Evaluating Evidence Attribution in Generated Fact Checking Explanations", "研究事实核查解释中的证据归因质量和引用精度。", 2024, "ACL", "DemoLocal", 0.82),
        ],
    },
    {
        "key": "vehicle_stability",
        "name": "车辆横摆稳定性与悬架控制",
        "patterns": [
            "vehicle active yaw control handling stability",
            "汽车悬架稳定性与舒适性",
            "车辆横摆稳定性",
            "悬架控制",
        ],
        "keywords": ["vehicle", "yaw", "control", "handling", "stability", "suspension", "汽车", "悬架", "稳定性", "舒适性"],
        "entities": ["vehicle stability", "active yaw control", "active suspension", "ride comfort"],
        "methods": ["model predictive control", "integrated chassis control", "active suspension"],
        "datasets": [],
        "intent": "检索车辆横摆稳定、悬架控制和操纵舒适性相关论文。",
        "overview": "该方向围绕车辆操纵稳定性与舒适性的协同优化，可展示系统对汽车工程专业问题的检索能力。",
        "themes": ["横摆稳定", "主动悬架", "底盘协同", "舒适性控制"],
        "gaps": ["建议后续补充实验车型、工况和控制器硬件平台约束。"],
        "next": ["integrated chassis control review", "active suspension handling stability", "vehicle yaw stability MPC"],
        "papers": [
            ("Improvement of Both Handling Stability and Ride Comfort of a Vehicle via Coupled Hydraulically Interconnected Suspension and Electronic Controlled Air Spring", "研究液压互联悬架与电子控制空气弹簧对操纵稳定性和乘坐舒适性的联合提升。", 2019, "Proceedings of the Institution of Mechanical Engineers Part D", "DemoLocal", 0.93),
            ("A Review on Various Control Strategies and Algorithms in Vehicle Suspension Systems", "综述车辆悬架系统控制策略与算法，覆盖半主动、主动和鲁棒控制。", 2023, "International Journal of Automotive and Mechanical Engineering", "DemoLocal", 0.88),
            ("Active Yaw Control for Vehicle Handling and Stability Enhancement", "分析主动横摆控制在提升车辆操纵稳定性中的建模、控制与实验方法。", 2023, "Vehicle System Dynamics", "DemoLocal", 0.85),
            ("Integrated Chassis Control for Handling Stability and Ride Comfort", "研究转向、制动和悬架子系统协同控制以兼顾稳定性和舒适性。", 2022, "Mechanical Systems and Signal Processing", "DemoLocal", 0.82),
        ],
    },
    {
        "key": "heavy_truck_brake",
        "name": "重卡制动力分配与稳定性",
        "patterns": [
            "brake force distribution control for heavy trucks stability",
            "重卡制动力分配",
            "商用车制动稳定性",
            "heavy truck braking",
        ],
        "keywords": ["brake", "force", "distribution", "heavy", "truck", "stability", "重卡", "制动", "分配"],
        "entities": ["brake force distribution", "heavy truck", "stability control", "electronic braking system"],
        "methods": ["EBS control", "load transfer modeling", "brake allocation"],
        "datasets": [],
        "intent": "检索重型商用车制动力分配、紧急制动和稳定性控制论文。",
        "overview": "该方向关注高载荷车辆在制动与转向耦合工况下的稳定性，是汽车工程演示场景中更贴近专业特色的主题。",
        "themes": ["制动力分配", "载荷转移", "EBS控制", "重车稳定性"],
        "gaps": ["建议补充多轴车辆、不同附着路面和实车试验条件。"],
        "next": ["heavy truck brake allocation", "commercial vehicle EBS stability", "load-sensitive braking control"],
        "papers": [
            ("Brake Force Distribution Control for Heavy Commercial Vehicles", "面向重型商用车制动稳定性，研究轴间制动力动态分配策略。", 2022, "Vehicle System Dynamics", "DemoLocal", 0.90),
            ("Electronic Braking System Control for Heavy Trucks Under Emergency Maneuvers", "分析紧急制动与转向耦合工况下的EBS控制策略。", 2021, "SAE Technical Paper", "DemoLocal", 0.86),
            ("Stability Control of Articulated Heavy Vehicles During Braking", "研究铰接式重型车辆制动时横摆稳定性和折叠风险控制。", 2020, "IEEE Transactions", "DemoLocal", 0.83),
            ("Load-sensitive Brake Allocation for Multi-axle Trucks", "考虑载荷转移和多轴约束的制动力分配方法。", 2024, "arXiv", "DemoLocal", 0.80),
        ],
    },
    {
        "key": "autonomous_driving",
        "name": "自动驾驶轨迹预测与规划",
        "patterns": [
            "autonomous driving trajectory prediction uncertainty multimodal planning",
            "智能汽车路径规划",
            "自动驾驶轨迹预测",
            "多模态不确定性",
        ],
        "keywords": ["autonomous", "driving", "trajectory", "prediction", "uncertainty", "planning", "自动驾驶", "轨迹", "预测", "规划"],
        "entities": ["autonomous driving", "trajectory prediction", "uncertainty", "planning"],
        "methods": ["motion transformer", "multi-modal prediction", "planning-oriented forecasting"],
        "datasets": ["Waymo Open Motion", "Argoverse"],
        "intent": "检索自动驾驶轨迹预测、交互建模和规划导向预测论文。",
        "overview": "该方向展示系统对智能汽车与AI交叉主题的多策略检索能力，重点在多智能体交互、轨迹多模态和规划安全。",
        "themes": ["轨迹预测", "多模态", "不确定性", "规划导向"],
        "gaps": ["建议进一步补充真实道路数据集和封闭环仿真评测。"],
        "next": ["motion transformer autonomous driving", "trajectory prediction uncertainty", "planning oriented forecasting"],
        "papers": [
            ("Wayformer: Motion Forecasting via Simple and Efficient Attention Networks", "使用注意力网络对多智能体轨迹进行预测，兼顾效率和多模态性。", 2023, "ICRA", "DemoLocal", 0.92),
            ("MTR: Motion Transformer for Autonomous Driving", "提出基于Transformer的运动预测框架，建模道路拓扑和多智能体交互。", 2022, "NeurIPS", "DemoLocal", 0.89),
            ("MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction", "通过多路径候选和信息融合提升自动驾驶行为预测准确性。", 2021, "ICRA", "DemoLocal", 0.85),
            ("Planning-oriented Autonomous Driving Trajectory Prediction", "强调预测结果对下游规划安全性和可执行性的影响。", 2024, "arXiv", "DemoLocal", 0.82),
        ],
    },
    {
        "key": "battery_thermal",
        "name": "电动汽车电池热管理安全",
        "patterns": [
            "battery thermal management electric vehicle safety",
            "电池热管理安全",
            "电动汽车热失控预警",
            "battery thermal runaway",
        ],
        "keywords": ["battery", "thermal", "management", "electric", "vehicle", "safety", "runaway", "电池", "热管理", "安全"],
        "entities": ["battery thermal management", "electric vehicle", "thermal runaway", "safety monitoring"],
        "methods": ["thermal runaway prediction", "liquid cooling", "data-driven monitoring"],
        "datasets": [],
        "intent": "检索电动汽车电池热管理、热失控预警和安全监测论文。",
        "overview": "该方向适合展示系统对新能源汽车安全问题的论文梳理能力，覆盖热管理结构优化、数据驱动预警和安全监测。",
        "themes": ["热管理", "热失控预警", "液冷优化", "安全监测"],
        "gaps": ["建议补充电池包结构参数、传感器布置和车端部署约束。"],
        "next": ["battery thermal management review", "thermal runaway prediction EV", "battery safety monitoring"],
        "papers": [
            ("A Review of Battery Thermal Management Systems for Electric Vehicles", "综述电动汽车电池热管理系统的空气冷却、液冷和相变材料方案。", 2023, "Applied Thermal Engineering", "DemoLocal", 0.92),
            ("Thermal Runaway Prediction and Safety Management for Lithium-ion Batteries", "研究锂电池热失控预测、早期预警和安全管理策略。", 2024, "Journal of Power Sources", "DemoLocal", 0.88),
            ("Data-driven Battery Safety Monitoring for Electric Vehicles", "结合传感器数据和机器学习进行电池安全状态监测。", 2022, "Energy AI", "DemoLocal", 0.84),
            ("Optimization of Liquid Cooling Plates for Electric Vehicle Battery Packs", "面向电池包液冷板结构优化，提高温度一致性和散热效率。", 2021, "Energy Conversion and Management", "DemoLocal", 0.80),
        ],
    },
]
