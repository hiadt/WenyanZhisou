from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .schema import normalize_title


@dataclass
class EvalExample:
    query: str
    gold_ids: Set[str]
    gold_items: List[Set[str]]
    raw: Dict[str, Any]


def load_jsonl(path: str | Path, limit: int | None = None) -> List[EvalExample]:
    rows: List[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold_items = extract_gold_items(obj)
            rows.append(
                EvalExample(
                    query=extract_query(obj),
                    gold_ids={key for item in gold_items for key in item},
                    gold_items=gold_items,
                    raw=obj,
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def extract_query(obj: Dict[str, Any]) -> str:
    for key in ["query", "question", "input", "instruction", "search_query", "user_query"]:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    messages = obj.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return str(m["content"])
    raise ValueError(f"Cannot extract query from keys: {list(obj.keys())}")


def extract_gold_items(obj: Dict[str, Any]) -> List[Set[str]]:
    """
    Return gold answers as a list of alias sets.

    PaSa RealScholarQuery stores answers as paper titles plus arXiv ids:
    {"answer": [...], "answer_arxiv_id": [...]}.  A retrieved paper should be
    counted as correct when any alias of the same gold paper matches, but the
    paper must not be double-counted just because both title and arXiv id match.
    """
    if "answer" in obj or "answer_arxiv_id" in obj:
        titles = obj.get("answer") if isinstance(obj.get("answer"), list) else []
        arxiv_ids = obj.get("answer_arxiv_id") if isinstance(obj.get("answer_arxiv_id"), list) else []
        out: List[Set[str]] = []
        total = max(len(titles), len(arxiv_ids))
        for idx in range(total):
            aliases: Set[str] = set()
            if idx < len(titles):
                aliases |= normalize_eval_aliases(titles[idx])
            if idx < len(arxiv_ids):
                aliases |= normalize_eval_aliases(arxiv_ids[idx])
            if aliases:
                out.append(aliases)
        return out

    return [{key} for key in extract_gold_ids(obj)]


def extract_gold_ids(obj: Dict[str, Any]) -> Set[str]:
    candidates = []
    for key in [
        "gold_ids",
        "gold_paper_ids",
        "relevant_ids",
        "relevant_papers",
        "positive_pids",
        "answer_paper_ids",
        "answer_arxiv_id",
        "answer",
        "paper_ids",
        "references",
        "reference",
    ]:
        if key in obj:
            candidates.append(obj[key])
    out: Set[str] = set()
    for val in candidates:
        out |= _ids_from_value(val)
    return out


def _ids_from_value(val: Any) -> Set[str]:
    out: Set[str] = set()
    if val is None:
        return out
    if isinstance(val, (str, int)):
        out |= normalize_eval_aliases(val)
    elif isinstance(val, list):
        for item in val:
            out |= _ids_from_value(item)
    elif isinstance(val, dict):
        for key in [
            "paper_id",
            "paperId",
            "id",
            "corpusid",
            "doi",
            "openalex_id",
            "semantic_scholar_id",
            "arxiv_id",
            "arxivId",
            "title",
        ]:
            if val.get(key):
                out |= normalize_eval_aliases(val[key])
    return out


def normalize_eval_aliases(value: Any) -> Set[str]:
    text = str(value or "").strip()
    if not text:
        return set()

    aliases = {text}
    lower = text.lower().strip()
    aliases.add(lower)

    for arxiv_id in extract_arxiv_ids(text):
        aliases.add(f"arxiv:{arxiv_id}")

    title_key = normalize_title(text)
    if title_key and not extract_arxiv_ids(text):
        aliases.add(f"title:{title_key}")

    if lower.startswith("doi:"):
        aliases.add(lower)
    elif lower.startswith("10."):
        aliases.add(f"doi:{lower}")

    return {alias for alias in aliases if alias}


def extract_arxiv_ids(text: str) -> Set[str]:
    out: Set[str] = set()
    for match in re.finditer(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text or "", flags=re.I):
        out.add(match.group(1).lower())
    return out
