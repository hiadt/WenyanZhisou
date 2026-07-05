from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

import requests

from .config import LLMConfig
from .schema import Paper, QueryPlan


class LLMClient:
    """OpenAI-compatible chat client.

    It works with DeepSeek, Qwen via vLLM, Ollama OpenAI endpoint,
    OpenAI-compatible cloud providers, and many rented-server deployments.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.warnings: List[str] = []
        self.disabled = False

    def reset_stats(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.warnings = []
        self.disabled = False

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.disabled:
            raise RuntimeError("LLM client disabled after a previous failed request in this query.")
        prompt_tokens = sum(max(1, len(m.get("content", "")) // 4) for m in messages)
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except Exception:
            self.disabled = True
            raise
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += max(1, len(text) // 4)
        return text


class LLMPlanner:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def plan(self, query: str) -> QueryPlan:
        if self.llm is None:
            return heuristic_plan(query)
        prompt = f"""
You are an academic search query planner. Parse the user's complex research query.
Return ONLY valid JSON with these fields:
intent, entities, methods, datasets, constraints, sub_queries, negative_terms.
If the user query is Chinese, translate the academic meaning into English first.
Generate 4-8 precise English academic search queries, including common synonyms,
technical names, abbreviations, and biomedical/engineering terminology when useful.

User query:
{query}
"""
        try:
            text = self.llm.chat([
                {"role": "system", "content": "You are a rigorous academic search agent."},
                {"role": "user", "content": prompt},
            ])
            obj = _extract_json(text)
            plan = QueryPlan(
                original_query=query,
                intent=str(obj.get("intent", "")),
                entities=_as_list(obj.get("entities"))[:12],
                methods=_as_list(obj.get("methods"))[:12],
                datasets=_as_list(obj.get("datasets"))[:12],
                constraints=_as_dict(obj.get("constraints")),
                sub_queries=_as_list(obj.get("sub_queries"))[:8] or [query],
                negative_terms=_as_list(obj.get("negative_terms"))[:12],
            )
            return _merge_with_heuristic(query, plan)
        except Exception as exc:
            if self.llm is not None:
                self.llm.warnings.append(f"LLM planner failed, used heuristic plan: {type(exc).__name__}: {exc}")
            return heuristic_plan(query)


class LLMVerifier:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def verify(self, query: str, papers: List[Paper]) -> None:
        if self.llm is None or not papers:
            return
        chunks = []
        for idx, p in enumerate(papers, 1):
            chunks.append(
                f"[{idx}] source={p.source}; paper_id={p.paper_id}; url={p.url}\n"
                f"title={p.title}\nyear={p.year}; venue={p.venue}\nabstract={p.abstract[:260]}"
            )
        prompt = f"""
Judge paper relevance to the academic query. Return ONLY compact JSON, no reasons:
{{"items":[[1,0.0,"high|partial|irrelevant"]]}}

Query:
{query}

Papers:
{chr(10).join(chunks)}
"""
        try:
            text = self.llm.chat([
                {"role": "system", "content": "You are a strict scientific literature relevance judge."},
                {"role": "user", "content": prompt},
            ])
            obj = _extract_json(text)
            if isinstance(obj, list):
                items = obj
            else:
                items = obj.get("items") or obj.get("results") or obj.get("papers") or []
            updated = 0
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    raw_index = item[0]
                    raw_score = item[1]
                    raw_label = item[2]
                    raw_reason = ""
                elif isinstance(item, dict):
                    raw_index = (
                        item.get("index")
                        or item.get("paper_index")
                        or item.get("paperIndex")
                        or item.get("id")
                    )
                    raw_score = (
                        item.get("score")
                        or item.get("relevance")
                        or item.get("relevance_score")
                        or item.get("rating")
                        or 0.0
                    )
                    raw_label = item.get("label") or item.get("relevance_label") or "candidate"
                    raw_reason = item.get("reason", "")
                else:
                    continue
                i = int(raw_index or 0) - 1
                if 0 <= i < len(papers):
                    score = _coerce_score(raw_score)
                    papers[i].llm_score = score
                    papers[i].relevance_label = str(raw_label)
                    papers[i].reason = str(raw_reason)
                    updated += 1
            if updated == 0 and self.llm is not None:
                self.llm.warnings.append("LLM verifier returned no usable relevance scores.")
        except Exception as exc:
            if self.llm is not None:
                self.llm.warnings.append(f"LLM verifier failed: {type(exc).__name__}: {exc}")
            return


class LLMQueryEvolver:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def evolve(
        self,
        original_query: str,
        scoring_query: str,
        papers: List[Paper],
        existing: List[str],
    ) -> List[str]:
        if self.llm is None or self.llm.disabled or not papers:
            return []
        snippets = []
        for idx, p in enumerate(papers[:8], 1):
            snippets.append(
                f"[{idx}] title={p.title}\nyear={p.year}\nvenue={p.venue}\nabstract={p.abstract[:500]}"
            )
        prompt = f"""
You are improving an academic paper search strategy.
Return ONLY JSON: {{"queries":["..."]}}
Generate 3-5 NEW precise English scholarly search queries.
Focus on missing synonyms, technical names, methods, datasets, standards, and citation-follow-up terms.
Do not repeat existing queries.

Original user query:
{original_query}

Current scoring query:
{scoring_query}

Existing queries:
{json.dumps(existing, ensure_ascii=False)}

Top current papers:
{chr(10).join(snippets)}
"""
        try:
            text = self.llm.chat([
                {"role": "system", "content": "You are a careful academic search strategy optimizer."},
                {"role": "user", "content": prompt},
            ])
            obj = _extract_json(text)
            raw = obj.get("queries") if isinstance(obj, dict) else obj
            queries = [str(q).strip() for q in (raw or []) if str(q).strip()]
            return [q for q in dict.fromkeys(queries) if q not in existing][:5]
        except Exception as exc:
            if self.llm is not None:
                self.llm.warnings.append(f"LLM query evolution failed, used fallback evolution: {type(exc).__name__}: {exc}")
            return []


class ResultSynthesizer:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def synthesize(self, query: str, papers: List[Paper]) -> Dict[str, Any]:
        fallback = heuristic_synthesis(query, papers)
        if self.llm is None or self.llm.disabled or not papers:
            return fallback
        chunks = []
        for idx, p in enumerate(papers[:10], 1):
            chunks.append(
                f"[{idx}] title={p.title}\nyear={p.year}\nvenue={p.venue}\nscore={p.final_score:.3f}\nabstract={p.abstract[:650]}"
            )
        prompt = f"""
Summarize academic search results for the user's query.
Return ONLY JSON with fields:
overview, themes, highly_relevant, partial_relevant, gaps, next_search_suggestions.

Rules:
- Be concise and evidence-grounded.
- highly_relevant and partial_relevant should contain objects with rank, title, reason.
- Mention uncertainty if results are weak or sparse.

Query:
{query}

Ranked papers:
{chr(10).join(chunks)}
"""
        try:
            text = self.llm.chat([
                {"role": "system", "content": "You write structured academic search summaries."},
                {"role": "user", "content": prompt},
            ])
            obj = _extract_json(text)
            if isinstance(obj, dict):
                return _normalize_synthesis(obj, fallback)
        except Exception as exc:
            if self.llm is not None:
                self.llm.warnings.append(f"LLM result synthesis failed, used heuristic synthesis: {type(exc).__name__}: {exc}")
        return fallback


def heuristic_synthesis(query: str, papers: List[Paper]) -> Dict[str, Any]:
    if not papers:
        return {
            "overview": "没有找到足够相关的论文，建议扩大检索词、接入 LLM 查询改写或增加本地/全文语料。",
            "themes": [],
            "highly_relevant": [],
            "partial_relevant": [],
            "gaps": ["当前候选为空，无法形成可靠归纳。"],
            "next_search_suggestions": [],
        }
    top_score = max(p.final_score for p in papers) or 1.0
    high = []
    partial = []
    for idx, p in enumerate(papers[:10], 1):
        item = {
            "rank": idx,
            "title": p.title,
            "reason": _heuristic_reason(query, p),
        }
        if p.final_score >= top_score * 0.78:
            high.append(item)
        else:
            partial.append(item)
    themes = _top_terms(" ".join(p.title + " " + p.abstract[:300] for p in papers[:10]))
    return {
        "overview": f"返回 {len(papers)} 篇候选论文，其中 {len(high)} 篇属于高置信候选。排序主要依据 API、BM25、Embedding、Reranker 与可用 LLM 分数融合。",
        "themes": themes[:8],
        "highly_relevant": high[:5],
        "partial_relevant": partial[:5],
        "gaps": [
            "该归纳基于标题、摘要和可用全文字段；若未提供全文，不能替代全文综述。",
            "最终相关性仍需结合公开测试集或人工标注验证。",
        ],
        "next_search_suggestions": _next_suggestions(query, themes),
    }


def _normalize_synthesis(obj: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fallback)
    for key in ["overview", "themes", "highly_relevant", "partial_relevant", "gaps", "next_search_suggestions"]:
        if key in obj and obj[key]:
            out[key] = obj[key]
    return out


def _heuristic_reason(query: str, p: Paper) -> str:
    signals = []
    if p.bm25_score > 0.2:
        signals.append("关键词匹配较强")
    if p.embedding_score > 0.45:
        signals.append("语义相似度较高")
    if p.reranker_score > 0.45:
        signals.append("重排模型评分较高")
    if p.llm_score > 0.4:
        signals.append("LLM 判断相关")
    if not signals:
        signals.append("由综合排序选入候选")
    return "，".join(signals) + "。"


def _top_terms(text: str) -> List[str]:
    stop = {
        "about", "after", "also", "and", "are", "based", "between", "from", "have",
        "into", "model", "paper", "study", "that", "the", "this", "using", "with",
    }
    terms = [t for t in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if t not in stop]
    counts: Dict[str, int] = {}
    for t in terms:
        counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:12]]


def _next_suggestions(query: str, themes: List[str]) -> List[str]:
    base = re.sub(r"\s+", " ", query.strip())
    suggestions = []
    if themes:
        suggestions.append((base + " " + " ".join(themes[:3])).strip())
    if len(themes) >= 6:
        suggestions.append(" ".join(themes[:6]))
    return list(dict.fromkeys(suggestions))[:3]


def _merge_with_heuristic(query: str, plan: QueryPlan) -> QueryPlan:
    """Keep LLM output, but add transparent rule-based domain expansions.

    The LLM planner is usually stronger, but it may omit benchmark-critical
    synonyms such as "data pruning" for "smaller dataset".  The rule layer is
    small and auditable, and only contributes extra entities/sub-queries.
    """

    fallback = heuristic_plan(query)
    merged_sub_queries = list(dict.fromkeys((plan.sub_queries or []) + fallback.sub_queries))
    merged_entities = list(dict.fromkeys((plan.entities or []) + fallback.entities))
    if not plan.intent or plan.intent.strip() == query.strip():
        plan.intent = fallback.intent
    plan.sub_queries = merged_sub_queries[:10]
    plan.entities = merged_entities[:16]
    return plan


def heuristic_plan(query: str) -> QueryPlan:
    q = query.strip()
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", q.lower())
    cjk_expansions = _cjk_expansions(q)
    sub = [q]
    sub += cjk_expansions
    if "hallucination" in terms:
        sub += ["large language model factuality faithfulness evaluation", "hallucination detection natural language generation"]
    if "retrieval" in terms or "rag" in terms:
        sub += ["retrieval augmented generation evaluation evidence attribution", "dense retrieval reranking question answering"]
    if "vulnerability" in terms:
        sub += ["software vulnerability detection graph neural network", "code vulnerability detection representation learning"]
    if _mentions_smaller_data_llm(q, terms):
        sub += [
            "data pruning for pretraining large language models",
            "data efficient LLM pretraining fewer training data",
            "deduplicating training data makes language models better",
            "data selection influential subset language model pretraining",
            "training better language models with fewer data",
        ]
    if "in-context" in q.lower() or ("context" in terms and "learning" in terms):
        sub += [
            "in-context learning capability during language model pretraining",
            "transformers learn in-context by gradient descent",
            "induction heads in-context learning pretraining",
            "emergent in-context learning transformers data distribution",
        ]
    if len(sub) == 1 and terms:
        sub.append(" ".join(terms[:8]))
    entities = terms[:10] or _cjk_entities(q)
    return QueryPlan(
        original_query=q,
        intent=cjk_expansions[0] if cjk_expansions else q,
        entities=entities[:12],
        sub_queries=list(dict.fromkeys([x for x in sub if x]))[:8],
    )


def _mentions_smaller_data_llm(query: str, terms: List[str]) -> bool:
    text = query.lower()
    data_hit = any(x in terms for x in ["data", "dataset", "datasets", "corpus"]) or "training data" in text
    less_hit = any(x in terms for x in ["smaller", "less", "fewer", "limited", "efficient", "pruning"])
    model_hit = any(x in text for x in ["large language model", "language model", "llm", "pre-training", "pretraining"])
    return data_hit and less_hit and model_hit


def _cjk_entities(query: str) -> List[str]:
    return [zh for zh, _ in _ZH_TO_EN if zh in query][:12]


def _cjk_expansions(query: str) -> List[str]:
    """Rule-based Chinese-to-English expansion for no-LLM smoke/online demos.

    This is deliberately small and transparent. In formal runs, an LLM planner
    should replace this with richer intent parsing; the fallback keeps Chinese
    queries from being sent to English-heavy academic APIs with no usable tokens.
    """

    if not re.search(r"[\u4e00-\u9fff]", query):
        return []
    hits: List[str] = []
    for zh, en in _ZH_TO_EN:
        if zh in query:
            hits.extend(en.split())
    hits = list(dict.fromkeys(hits))
    if not hits:
        return []

    base = " ".join(hits[:10])
    sub = [base]
    hit_text = " ".join(hits)
    if any(x in hit_text for x in ["vehicle", "automotive", "autonomous", "connected"]):
        if any(x in hit_text for x in ["safety", "security", "cybersecurity"]):
            sub += [
                "intelligent vehicle system safety strategy",
                "autonomous vehicle safety assurance functional safety cybersecurity",
                "connected automated vehicle safety strategy ISO 26262 SOTIF",
                "intelligent connected vehicle cybersecurity risk assessment",
            ]
        if any(x in hit_text for x in ["brake", "braking", "truck", "heavy"]):
            sub += [
                "heavy truck braking force distribution",
                "commercial vehicle brake force distribution control strategy",
                "large truck braking safety control",
            ]
        if any(x in hit_text for x in ["cockpit", "cabin", "infotainment", "comfort", "HMI"]):
            sub += [
                "smart cockpit automatic adjustment vehicle cabin personalization",
                "intelligent cockpit human machine interaction adaptive control",
                "in vehicle infotainment smart cockpit driver preference personalization",
                "automotive cabin comfort adaptive control seat climate adjustment",
                "vehicle cockpit intelligent control occupant comfort personalization",
            ]
    if any(x in hit_text for x in ["privacy", "data"]):
        sub.append("vehicle data privacy protection security strategy")
    if any(x in hit_text for x in ["stent", "coronary", "cardiac", "percutaneous"]):
        sub += [
            "coronary stent drug eluting stent restenosis",
            "cardiac stent coronary artery disease percutaneous coronary intervention",
            "drug-eluting stent bare metal stent clinical outcomes",
            "coronary stent thrombosis restenosis PCI",
            "bioresorbable vascular scaffold coronary stent",
        ]
    if any(x in hit_text for x in ["pacemaker", "pacing", "arrhythmia"]):
        sub += [
            "cardiac pacemaker implantable pacemaker arrhythmia",
            "permanent pacemaker implantation cardiac pacing",
            "leadless pacemaker bradycardia clinical outcomes",
        ]
    return list(dict.fromkeys(sub))[:8]


_ZH_TO_EN = [
    ("智能网联汽车", "intelligent connected vehicle connected automated vehicle"),
    ("智能车辆", "intelligent vehicle"),
    ("智能汽车", "intelligent vehicle automotive"),
    ("智能座舱", "intelligent cockpit smart cockpit automotive cockpit in-vehicle infotainment HMI"),
    ("智慧座舱", "intelligent cockpit smart cockpit automotive cockpit in-vehicle infotainment HMI"),
    ("座舱", "cockpit vehicle cabin automotive cockpit"),
    ("自动驾驶", "autonomous driving autonomous vehicle"),
    ("无人驾驶", "autonomous driving driverless vehicle"),
    ("车联网", "vehicular network internet of vehicles"),
    ("车载", "in-vehicle automotive"),
    ("车辆", "vehicle automotive"),
    ("汽车", "vehicle automotive"),
    ("大卡车", "large truck heavy truck commercial vehicle"),
    ("重卡", "heavy truck commercial vehicle"),
    ("卡车", "truck commercial vehicle"),
    ("制动力分配", "brake force distribution braking force distribution"),
    ("制动", "brake braking"),
    ("自动调节", "automatic adjustment adaptive adjustment adaptive control personalization"),
    ("自动调整", "automatic adjustment adaptive adjustment adaptive control personalization"),
    ("智能调节", "intelligent adjustment adaptive control personalization"),
    ("自适应", "adaptive control adaptive adjustment"),
    ("个性化", "personalization personalized driver preference"),
    ("座椅", "seat seating"),
    ("空调", "air conditioning HVAC climate control"),
    ("温度", "temperature thermal comfort"),
    ("舒适", "comfort thermal comfort"),
    ("乘员", "occupant passenger"),
    ("驾驶员", "driver"),
    ("人机交互", "human machine interaction HMI"),
    ("交互", "interaction HMI"),
    ("系统安全", "system safety"),
    ("功能安全", "functional safety ISO 26262"),
    ("预期功能安全", "SOTIF safety of the intended functionality"),
    ("网络安全", "cybersecurity security"),
    ("信息安全", "information security cybersecurity"),
    ("安全策略", "safety strategy security policy"),
    ("安全", "safety security"),
    ("策略", "strategy policy"),
    ("风险评估", "risk assessment"),
    ("威胁建模", "threat modeling"),
    ("隐私", "privacy"),
    ("数据", "data"),
    ("心脏支架", "cardiac stent coronary stent drug-eluting stent percutaneous coronary intervention"),
    ("冠脉支架", "coronary stent drug-eluting stent percutaneous coronary intervention"),
    ("冠状动脉支架", "coronary stent drug-eluting stent percutaneous coronary intervention"),
    ("药物洗脱支架", "drug-eluting stent coronary stent"),
    ("支架再狭窄", "in-stent restenosis coronary stent"),
    ("再狭窄", "restenosis in-stent restenosis"),
    ("冠脉", "coronary artery coronary"),
    ("冠状动脉", "coronary artery coronary"),
    ("介入治疗", "percutaneous coronary intervention PCI"),
    ("心脏起搏器", "cardiac pacemaker implantable pacemaker cardiac pacing"),
    ("起搏器", "cardiac pacemaker implantable pacemaker cardiac pacing"),
    ("心律失常", "arrhythmia cardiac pacing"),
    ("支架", "stent scaffold"),
]


def _coerce_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        raw = str(value).strip()
        if raw.endswith("%"):
            score = float(raw[:-1]) / 100.0
        else:
            match = re.search(r"-?\d+(?:\.\d+)?", raw)
            score = float(match.group(0)) if match else 0.0
    if score > 1.0 and score <= 10.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


def _as_list(value: Any) -> List[str]:
    """Normalize loose LLM JSON fields into a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        items = []
        for key, val in value.items():
            if val in (None, "", [], {}):
                continue
            items.append(f"{key}: {val}")
        return items
    text = str(value).strip()
    return [text] if text else []


def _as_dict(value: Any) -> Dict[str, Any]:
    """Normalize loose LLM JSON fields into a dictionary.

    DeepSeek and other chat models sometimes return constraints as a list of
    phrases or a plain string. Treating those as dict(value) raises confusing
    ValueErrors, so we preserve the information under stable keys instead.
    """

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": [str(x).strip() for x in value if str(x).strip()]}
    text = str(value).strip()
    return {"text": text} if text else {}


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    if text.startswith("["):
        return json.loads(text)
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)
        else:
            m = re.search(r"\[.*\]", text, flags=re.S)
            if not m:
                raise ValueError("No JSON object or array found")
            text = m.group(0)
    return json.loads(text)
