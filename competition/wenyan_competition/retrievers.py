from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import quote

import requests

from .config import RetrievalConfig
from .schema import Paper


class AcademicRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.api_calls = 0
        self.warnings: List[str] = []
        self._local_retriever = (
            LocalCorpusRetriever(
                self.config.local_corpus_path,
                self.config.per_query,
                min_score=self.config.local_min_score,
            )
            if self.config.local_corpus_path
            else None
        )
        self._pasa_retriever = PasaTitleRetriever.from_path(
            self.config.pasa_id2paper_path,
            limit=self.config.pasa_title_limit,
            min_score=self.config.pasa_title_min_score,
        )

    def reset_stats(self) -> None:
        self.api_calls = 0
        self.warnings = []

    def search_many(self, queries: Iterable[str]) -> List[Paper]:
        query_list = list(queries)
        papers: List[Paper] = []
        if self._local_retriever:
            for q in query_list:
                papers.extend(self._local_retriever.search(q))
        if self._pasa_retriever:
            for q in query_list:
                papers.extend(self._pasa_retriever.search(q))
        for q in query_list:
            if self.config.use_serper and self.config.serper_api_key:
                papers.extend(self._safe(self.search_serper_arxiv, q))
            if self.config.use_arxiv:
                papers.extend(self._safe(self.search_arxiv, q))
            if self.config.use_openalex:
                papers.extend(self._safe(self.search_openalex, q))
            if self.config.use_semantic_scholar:
                papers.extend(self._safe(self.search_semantic_scholar, q))
        return deduplicate(papers)

    def expand_citation_network(self, seeds: List[Paper], max_api_calls: int) -> List[Paper]:
        """Fetch one-hop reference/citation papers from high-value seeds."""

        ids: List[str] = []
        for seed in seeds[: self.config.citation_expand_seeds]:
            ids.extend(seed.references or [])
            ids.extend(seed.citations or [])
        ids = list(dict.fromkeys(x for x in ids if x))[: self.config.citation_expand_limit]

        out: List[Paper] = []
        for paper_id in ids:
            if self.api_calls >= max_api_calls:
                break
            if self.config.use_openalex and "openalex.org" in paper_id:
                out.extend(self._safe(self.fetch_openalex_work, paper_id, warn=False))
            elif self.config.use_semantic_scholar:
                out.extend(self._safe(self.fetch_semantic_scholar_paper, paper_id, warn=False))
        return deduplicate(out)

    def search_openalex(self, query: str) -> List[Paper]:
        self.api_calls += 1
        params = [
            "search=" + quote(query),
            f"per-page={min(self.config.per_query, 50)}",
            "sort=relevance_score:desc",
        ]
        if self.config.openalex_mailto:
            params.append("mailto=" + quote(self.config.openalex_mailto))
        url = "https://api.openalex.org/works?" + "&".join(params)
        data = requests.get(url, timeout=20).json()
        out: List[Paper] = []
        for item in data.get("results", []):
            title = item.get("title") or item.get("display_name") or ""
            if not title:
                continue
            venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "")
            publication_type = str(item.get("type") or item.get("type_crossref") or "")
            abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
            if self.config.academic_only and _looks_non_academic(
                title=title,
                abstract=abstract,
                venue=venue,
                publication_type=publication_type,
            ):
                continue
            out.append(
                Paper(
                    paper_id=item.get("id", ""),
                    title=title,
                    abstract=abstract,
                    full_text="",
                    year=item.get("publication_year"),
                    authors=[
                        a.get("author", {}).get("display_name", "")
                        for a in item.get("authorships", [])
                        if a.get("author", {}).get("display_name")
                    ],
                    venue=venue,
                    doi=(item.get("doi") or "").replace("https://doi.org/", ""),
                    url=((item.get("primary_location") or {}).get("landing_page_url") or item.get("id") or ""),
                    citation_count=int(item.get("cited_by_count") or 0),
                    source="OpenAlex",
                    publication_type=publication_type,
                    references=list(item.get("referenced_works") or []),
                    citations=list(item.get("related_works") or []),
                    api_score=float(item.get("relevance_score") or 0.0),
                )
            )
        return out

    def search_arxiv(self, query: str) -> List[Paper]:
        out: List[Paper] = []
        seen = set()
        for search_query in _arxiv_queries(query):
            self.api_calls += 1
            params = {
                "search_query": search_query,
                "start": "0",
                "max_results": str(min(self.config.per_query, 50)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            resp = requests.get("https://export.arxiv.org/api/query", params=params, timeout=20)
            if resp.status_code >= 400:
                continue
            root = ET.fromstring(resp.text)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            for idx, entry in enumerate(entries):
                paper = _paper_from_arxiv_entry(entry, idx)
                if not paper.title or paper.paper_id in seen:
                    continue
                seen.add(paper.paper_id)
                out.append(paper)
        return out

    def search_serper_arxiv(self, query: str) -> List[Paper]:
        """PaSa-style crawler: Google-like search constrained to arxiv.org.

        PaSa gets much of its RealScholarQuery coverage by first finding arXiv
        pages through web search and only then letting the selector decide which
        papers matter.  OpenAlex/S2/arXiv native search often miss long,
        natural-language requests, so this optional source adds that missing
        crawler behavior when SERPER_API_KEY is configured.
        """

        if not self.config.serper_api_key:
            return []
        found_ids: List[str] = []
        headers = {
            "X-API-KEY": self.config.serper_api_key,
            "Content-Type": "application/json",
        }
        for search_query in _serper_arxiv_queries(query):
            self.api_calls += 1
            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json={"q": search_query, "num": min(max(1, self.config.serper_top_k), 20)},
                timeout=20,
            )
            if resp.status_code >= 400:
                self.warnings.append(f"search_serper_arxiv failed: HTTP {resp.status_code}")
                continue
            try:
                found_ids.extend(_extract_arxiv_ids_from_serper(resp.json()))
            except ValueError as exc:
                self.warnings.append(f"search_serper_arxiv failed to parse JSON: {exc}")
        found_ids = list(dict.fromkeys(found_ids))[: self.config.serper_arxiv_limit]
        return self._fetch_arxiv_ids(found_ids, source="SerperArxiv")

    def _fetch_arxiv_ids(self, arxiv_ids: List[str], source: str = "arXivID") -> List[Paper]:
        ids = [x for x in list(dict.fromkeys(_normal_arxiv_id(x) for x in arxiv_ids)) if x]
        if not ids:
            return []
        out: List[Paper] = []
        for batch in _batches(ids, 20):
            self.api_calls += 1
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": ",".join(batch), "max_results": str(len(batch))},
                timeout=20,
            )
            if resp.status_code >= 400:
                continue
            root = ET.fromstring(resp.text)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            for idx, entry in enumerate(entries):
                paper = _paper_from_arxiv_entry(entry, idx)
                if not paper.title:
                    continue
                paper.source = source
                paper.api_score = max(paper.api_score, 0.95 - idx * 0.01)
                out.append(paper)
        return out

    def fetch_openalex_work(self, work_id: str) -> List[Paper]:
        self.api_calls += 1
        url = _openalex_api_work_url(work_id)
        item = _get_json_or_none(url)
        if not item:
            return []
        title = item.get("title") or item.get("display_name") or ""
        if not title:
            return []
        venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        publication_type = str(item.get("type") or item.get("type_crossref") or "")
        abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
        if self.config.academic_only and _looks_non_academic(
            title=title,
            abstract=abstract,
            venue=venue,
            publication_type=publication_type,
        ):
            return []
        return [
            Paper(
                paper_id=item.get("id", ""),
                title=title,
                abstract=abstract,
                full_text="",
                year=item.get("publication_year"),
                authors=[
                    a.get("author", {}).get("display_name", "")
                    for a in item.get("authorships", [])
                    if a.get("author", {}).get("display_name")
                ],
                venue=venue,
                doi=(item.get("doi") or "").replace("https://doi.org/", ""),
                url=((item.get("primary_location") or {}).get("landing_page_url") or item.get("id") or ""),
                citation_count=int(item.get("cited_by_count") or 0),
                source="OpenAlexCitation",
                publication_type=publication_type,
                references=list(item.get("referenced_works") or []),
                citations=list(item.get("related_works") or []),
                api_score=float(item.get("relevance_score") or 0.0),
            )
        ]

    def search_semantic_scholar(self, query: str) -> List[Paper]:
        self.api_calls += 1
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "venue",
                "url",
                "citationCount",
                "externalIds",
                "publicationTypes",
                "publicationVenue",
                "journal",
                "references.paperId",
                "citations.paperId",
            ]
        )
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={quote(query)}&limit={min(self.config.per_query, 100)}&fields={quote(fields)}"
        )
        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key
        data = requests.get(url, headers=headers, timeout=20).json()
        out: List[Paper] = []
        for item in data.get("data", []):
            title = item.get("title") or ""
            if not title:
                continue
            external = item.get("externalIds") or {}
            publication_types = item.get("publicationTypes") or []
            publication_type = "/".join(str(x) for x in publication_types if x)
            publication_venue = item.get("publicationVenue") or {}
            journal = item.get("journal") or {}
            venue = (
                item.get("venue")
                or publication_venue.get("name")
                or journal.get("name")
                or ""
            )
            abstract = item.get("abstract") or ""
            if self.config.academic_only and _looks_non_academic(
                title=title,
                abstract=abstract,
                venue=venue,
                publication_type=publication_type,
            ):
                continue
            out.append(
                Paper(
                    paper_id=item.get("paperId", ""),
                    title=title,
                    abstract=abstract,
                    full_text="",
                    year=item.get("year"),
                    authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                    venue=venue,
                    doi=external.get("DOI", ""),
                    url=item.get("url") or "",
                    citation_count=int(item.get("citationCount") or 0),
                    source="SemanticScholar",
                    publication_type=publication_type,
                    references=[x.get("paperId", "") for x in (item.get("references") or []) if x.get("paperId")],
                    citations=[x.get("paperId", "") for x in (item.get("citations") or []) if x.get("paperId")],
                    api_score=1.0,
                )
            )
        return out

    def fetch_semantic_scholar_paper(self, paper_id: str) -> List[Paper]:
        self.api_calls += 1
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "venue",
                "url",
                "citationCount",
                "externalIds",
                "publicationTypes",
                "publicationVenue",
                "journal",
                "references.paperId",
                "citations.paperId",
            ]
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id, safe='')}?fields={quote(fields)}"
        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key
        item = _get_json_or_none(url, headers=headers)
        if not item:
            return []
        title = item.get("title") or ""
        if not title:
            return []
        external = item.get("externalIds") or {}
        publication_types = item.get("publicationTypes") or []
        publication_type = "/".join(str(x) for x in publication_types if x)
        publication_venue = item.get("publicationVenue") or {}
        journal = item.get("journal") or {}
        venue = item.get("venue") or publication_venue.get("name") or journal.get("name") or ""
        abstract = item.get("abstract") or ""
        if self.config.academic_only and _looks_non_academic(
            title=title,
            abstract=abstract,
            venue=venue,
            publication_type=publication_type,
        ):
            return []
        return [
            Paper(
                paper_id=item.get("paperId", ""),
                title=title,
                abstract=abstract,
                full_text="",
                year=item.get("year"),
                authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                venue=venue,
                doi=external.get("DOI", ""),
                url=item.get("url") or "",
                citation_count=int(item.get("citationCount") or 0),
                source="SemanticScholarCitation",
                publication_type=publication_type,
                references=[x.get("paperId", "") for x in (item.get("references") or []) if x.get("paperId")],
                citations=[x.get("paperId", "") for x in (item.get("citations") or []) if x.get("paperId")],
                api_score=0.8,
            )
        ]

    def _safe(self, fn, query: str, warn: bool = True) -> List[Paper]:
        try:
            return fn(query)
        except Exception as exc:
            if warn:
                self.warnings.append(f"{fn.__name__} failed for query '{query}': {exc}")
            return []


def deduplicate(papers: List[Paper]) -> List[Paper]:
    by_key: Dict[str, Paper] = {}
    for p in papers:
        key = p.key()
        if key not in by_key:
            by_key[key] = p
            continue
        old = by_key[key]
        if len(p.abstract) > len(old.abstract):
            old.abstract = p.abstract
        old.citation_count = max(old.citation_count, p.citation_count)
        old.references = list(dict.fromkeys(old.references + p.references))
        old.citations = list(dict.fromkeys(old.citations + p.citations))
        if p.source not in old.source:
            old.source += "+" + p.source
    return list(by_key.values())


class LocalCorpusRetriever:
    """Small offline retriever for smoke tests and private corpora.

    The corpus can be JSONL or JSON list. It accepts fields commonly seen in
    academic datasets: paper_id/id, title, abstract, year, authors, venue,
    doi, url, citation_count, references, citations.
    """

    def __init__(self, path: str, limit: int = 20, min_score: float = 0.0):
        self.path = path
        self.limit = limit
        self.min_score = min_score
        self._papers = self._load()

    def search(self, query: str) -> List[Paper]:
        q_terms = set(_query_tokens(query))
        if not q_terms:
            return []
        scored = []
        for p in self._papers:
            terms = set(_tokens(p.text()))
            overlap = len(q_terms & terms) / max(1, len(q_terms))
            citation_bonus = min(1.0, (p.citation_count or 0) / 500.0) * 0.08
            score = overlap + citation_bonus
            if score >= self.min_score:
                scored.append((score, replace(p, api_score=score)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[: self.limit]]

    def _load(self) -> List[Paper]:
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        papers = []
        for row in rows:
            papers.append(
                Paper(
                    paper_id=str(row.get("paper_id") or row.get("id") or row.get("corpusid") or ""),
                    title=str(row.get("title") or ""),
                    abstract=str(row.get("abstract") or row.get("abstractText") or ""),
                    full_text=str(
                        row.get("full_text")
                        or row.get("fullText")
                        or row.get("text")
                        or row.get("body")
                        or ""
                    ),
                    year=row.get("year"),
                    authors=row.get("authors") or [],
                    venue=str(row.get("venue") or row.get("source") or ""),
                    doi=str(row.get("doi") or ""),
                    url=str(row.get("url") or ""),
                    citation_count=int(row.get("citation_count") or row.get("citationCount") or 0),
                    source="LocalCorpus",
                    references=row.get("references") or [],
                    citations=row.get("citations") or [],
                )
            )
        return papers


class PasaTitleRetriever:
    """High-recall title retriever for PaSa's local arXiv paper database.

    PaSa publishes a large `paper_database/id2paper.json` mapping arXiv ids to
    paper titles.  OpenAlex/Semantic Scholar do not always return those exact
    arXiv papers for long natural-language queries, so this lightweight inverted
    index uses the local title database as an additional candidate source.  The
    normal ranker still decides the final order.
    """

    def __init__(self, path: str, limit: int = 80, min_score: float = 0.10):
        self.path = path
        self.limit = limit
        self.min_score = min_score
        self._papers: List[Paper] = []
        self._title_terms: List[set[str]] = []
        self._index: Dict[str, List[int]] = {}
        self._df: Counter[str] = Counter()
        self._load()

    @classmethod
    def from_path(cls, path: str, limit: int = 80, min_score: float = 0.10):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return cls(str(p), limit=limit, min_score=min_score)

    def search(self, query: str) -> List[Paper]:
        query_terms = _expanded_pasa_query_terms(query)
        if not query_terms:
            return []
        qset = set(query_terms)
        query_idf = {term: self._idf(term) for term in qset}
        counts: Counter[int] = Counter()
        for term in qset:
            for idx in self._index.get(term, []):
                counts[idx] += 1
        if not counts:
            return []

        scored = []
        max_scan = max(self.limit * 40, self.limit)
        query_weight = sum(query_idf.values()) or 1.0
        for idx, raw_count in counts.most_common(max_scan):
            title_terms = self._title_terms[idx]
            if not title_terms:
                continue
            matched = qset & title_terms
            idf_hit = sum(query_idf[t] for t in matched)
            idf_coverage = idf_hit / query_weight
            density = len(matched) / max(1, min(len(title_terms), 14))
            score = 0.72 * idf_coverage + 0.20 * density
            score += _pasa_title_concept_bonus(query, self._papers[idx].title)
            if raw_count >= 4:
                score += 0.04
            if score >= self.min_score:
                scored.append((score, self._papers[idx]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [replace(p, api_score=min(1.0, score)) for score, p in scored[: self.limit]]

    def _load(self) -> None:
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        for arxiv_id, value in data.items():
            if isinstance(value, dict):
                title = str(value.get("title") or value.get("name") or "")
                year = value.get("year")
            else:
                title = str(value or "")
                year = None
            if not title:
                continue
            idx = len(self._papers)
            paper = Paper(
                paper_id=str(arxiv_id),
                title=title,
                doi=f"10.48550/arXiv.{arxiv_id}",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                year=year,
                source="PaSaTitleDB",
                publication_type="preprint",
            )
            terms = set(_pasa_tokens(title))
            self._papers.append(paper)
            self._title_terms.append(terms)
            self._df.update(terms)
            for term in terms:
                self._index.setdefault(term, []).append(idx)

    def _idf(self, term: str) -> float:
        total = max(1, len(self._papers))
        return math.log((total + 1) / (self._df.get(term, 0) + 1)) + 1.0


def _openalex_abstract(inv: Dict[str, List[int]]) -> str:
    pos = {}
    for word, indexes in inv.items():
        for i in indexes:
            pos[int(i)] = word
    return " ".join(pos[i] for i in sorted(pos))


def _openalex_api_work_url(work_id: str) -> str:
    work_id = str(work_id or "").strip()
    match = re.search(r"(W\d+)$", work_id)
    if match:
        return "https://api.openalex.org/works/" + match.group(1)
    if work_id.startswith("https://api.openalex.org/works/"):
        return work_id
    return "https://api.openalex.org/works/" + quote(work_id, safe="")


def _serper_arxiv_queries(query: str) -> List[str]:
    q = re.sub(r"\s+", " ", str(query or "")).strip()
    q_lower = q.lower()
    queries: List[str] = []

    if _mentions_smaller_data(q_lower):
        queries.extend(
            [
                "data pruning pretraining LLM smaller dataset site:arxiv.org/abs",
                "data efficient large language model pretraining site:arxiv.org/abs",
                "deduplicating training data language models site:arxiv.org/abs",
            ]
        )
    if "in-context" in q_lower or "in context" in q_lower:
        queries.extend(
            [
                "in-context learning pretraining transformers site:arxiv.org/abs",
                "emergence in-context learning language models site:arxiv.org/abs",
            ]
        )
    if q:
        queries.append(f"{q} site:arxiv.org/abs")
    terms = _pasa_tokens(q)
    focused = [t for t in terms if t not in {"give", "show", "result", "results", "paper", "papers"}][:12]
    if focused:
        queries.append(f"{' '.join(focused)} site:arxiv.org/abs")
    return list(dict.fromkeys(x for x in queries if x))[:3]


def _extract_arxiv_ids_from_serper(data) -> List[str]:
    text = json.dumps(data or {}, ensure_ascii=False)
    ids = [_normal_arxiv_id(m.group(0)) for m in re.finditer(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)\s*[\w.\-/]+", text, re.I)]
    ids.extend(_normal_arxiv_id(m.group(0)) for m in re.finditer(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text, re.I))
    return [x for x in list(dict.fromkeys(ids)) if x]


def _normal_arxiv_id(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", text, flags=re.I)
    if match:
        return match.group(1)
    match = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", text, flags=re.I)
    return match.group(1) if match else ""


def _batches(items: List[str], size: int):
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]


def _arxiv_queries(query: str) -> List[str]:
    q = (query or "").strip()
    q_lower = q.lower()
    queries: List[str] = []

    if _mentions_smaller_data(q_lower):
        queries.extend(
            [
                "all:data AND all:pruning AND all:pretraining AND all:llm",
                "all:data-efficient AND all:llm",
                "all:deduplicating AND all:training AND all:data AND all:language AND all:models",
            ]
        )
    if "in-context" in q_lower or "in context" in q_lower:
        queries.extend(
            [
                "all:in-context AND all:learning AND all:pretraining",
                "all:transformers AND all:in-context AND all:learning",
                "all:induction AND all:heads AND all:in-context",
            ]
        )
    terms = _pasa_tokens(q)
    if terms:
        focused = [t for t in terms if t not in {"give", "show", "result", "better"}][:7]
        if focused:
            queries.append(" AND ".join(f"all:{_arxiv_escape(t)}" for t in focused))
    if q:
        queries.append("all:" + _arxiv_escape(" ".join(terms[:10]) if terms else q[:180]))

    return list(dict.fromkeys(x for x in queries if x))[:3]


def _arxiv_escape(text: str) -> str:
    return re.sub(r"\s+", "+", str(text or "").strip())


def _paper_from_arxiv_entry(entry, rank: int) -> Paper:
    title = _clean_arxiv_text(entry.findtext("{http://www.w3.org/2005/Atom}title", ""))
    abstract = _clean_arxiv_text(entry.findtext("{http://www.w3.org/2005/Atom}summary", ""))
    url = entry.findtext("{http://www.w3.org/2005/Atom}id", "") or ""
    arxiv_id = _arxiv_id_from_url(url)
    authors = [
        _clean_arxiv_text(a.findtext("{http://www.w3.org/2005/Atom}name", ""))
        for a in entry.findall("{http://www.w3.org/2005/Atom}author")
    ]
    authors = [a for a in authors if a]
    published = entry.findtext("{http://www.w3.org/2005/Atom}published", "") or ""
    doi = entry.findtext("{http://arxiv.org/schemas/atom}doi", "") or ""
    if not doi and arxiv_id:
        doi = f"10.48550/arXiv.{arxiv_id}"
    return Paper(
        paper_id=arxiv_id,
        title=title,
        abstract=abstract,
        year=_year_from_date(published),
        authors=authors,
        venue="arXiv",
        doi=doi,
        url=url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
        source="arXiv",
        publication_type="preprint",
        api_score=max(0.0, 1.0 - rank * 0.02),
    )


def _arxiv_id_from_url(url: str) -> str:
    match = re.search(r"/abs/([^/?#]+)", url or "")
    raw = match.group(1) if match else str(url or "")
    raw = raw.strip().split("/")[-1]
    return re.sub(r"v\d+$", "", raw, flags=re.I)


def _clean_arxiv_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _year_from_date(text: str):
    match = re.search(r"\b(19|20)\d{2}\b", text or "")
    return int(match.group(0)) if match else None


def _get_json_or_none(url: str, headers: Dict[str, str] | None = None):
    resp = requests.get(url, headers=headers or {}, timeout=20)
    if resp.status_code >= 400:
        return None
    content_type = resp.headers.get("content-type", "").lower()
    if "json" not in content_type and not resp.text.lstrip().startswith(("{", "[")):
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())


def _pasa_tokens(text: str) -> List[str]:
    out: List[str] = []
    for token in _tokens(text):
        if token in _QUERY_STOPWORDS:
            continue
        out.append(token)
        if "-" in token:
            out.append(token.replace("-", ""))
            out.extend(part for part in token.split("-") if len(part) >= 3 and part not in _QUERY_STOPWORDS)
        if token.endswith("s") and len(token) > 4:
            out.append(token[:-1])
    return list(dict.fromkeys(out))


def _expanded_pasa_query_terms(query: str) -> List[str]:
    q = (query or "").lower()
    terms = set(_pasa_tokens(query))

    if "large language model" in q or "language models" in q or "llm" in q:
        terms.update(["llm", "llms", "language", "model", "models", "transformer", "transformers"])
    if "pre-training" in q or "pretraining" in q or "pre train" in q:
        terms.update(["pretraining", "pretrain", "training", "train"])
    if "in-context" in q or "in context" in q:
        terms.update(["in-context", "incontext", "context", "icl", "induction", "heads"])
    if _mentions_smaller_data(q):
        terms.update(
            [
                "data",
                "dataset",
                "datasets",
                "pruning",
                "selection",
                "selecting",
                "deduplicating",
                "deduplication",
                "efficient",
                "data-efficient",
                "dataefficient",
                "less",
                "fewer",
                "influential",
                "subset",
                "distilling",
                "alpaca",
                "alpagasus",
                "instruction",
                "tuning",
            ]
        )
    return [t for t in terms if t and t not in _QUERY_STOPWORDS]


def _mentions_smaller_data(query: str) -> bool:
    data_hit = any(x in query for x in ["data", "dataset", "datasets", "corpus", "pre-training", "pretraining"])
    less_hit = any(x in query for x in ["smaller", "small", "less", "fewer", "limited", "efficient", "pruning", "selection"])
    model_hit = any(x in query for x in ["llm", "language model", "pre-training", "pretraining", "training"])
    return data_hit and less_hit and model_hit


def _pasa_title_concept_bonus(query: str, title: str) -> float:
    q = (query or "").lower()
    t = (title or "").lower()
    bonus = 0.0
    if _mentions_smaller_data(q):
        phrase_groups = [
            ["less is more", "less:", "fewer data", "smaller dataset", "limited data"],
            ["data pruning", "data selection", "document selection", "influential subset", "deduplicat"],
            ["data-efficient", "efficient llm", "efficient encoder", "pretraining", "pre-training"],
            ["distilling", "step-by-step", "alpaca", "alpagasus", "babyllama"],
        ]
        hits = sum(1 for group in phrase_groups if any(x in t for x in group))
        bonus += min(0.22, hits * 0.055)
    if "in-context" in q or "in context" in q:
        phrase_groups = [
            ["in-context", "in context"],
            ["gradient descent", "induction heads", "bayesian", "kernel regression"],
            ["transformers", "pretraining", "pre-training", "emergent"],
        ]
        hits = sum(1 for group in phrase_groups if any(x in t for x in group))
        bonus += min(0.18, hits * 0.06)
    return bonus


_BAD_PUBLICATION_TYPES = {
    "news",
    "newspaper-article",
    "magazine-article",
    "posted-content",
}

_BAD_VENUE_PATTERNS = [
    "newspaper",
    "magazine",
    "weekly",
    "daily",
    "news",
    "gazette",
    "herald",
    "齐鲁周刊",
    "周刊",
    "日报",
    "晚报",
    "晨报",
    "时报",
    "商报",
    "都市报",
    "新闻",
]

_BAD_TITLE_PATTERNS = [
    "记者",
    "通讯员",
    "本报",
    "日讯",
    "专访",
    "商业裂变",
    "倒下前",
]


def _looks_non_academic(
    *,
    title: str,
    abstract: str,
    venue: str,
    publication_type: str,
) -> bool:
    """Conservatively drop obvious news/magazine-like hits.

    We avoid filtering just because a paper has no DOI or abstract, since many
    real scholarly APIs have sparse metadata. The rules target clear newspaper
    and magazine signals that repeatedly polluted Chinese demo queries.
    """

    ptype = (publication_type or "").lower()
    if any(t in ptype for t in _BAD_PUBLICATION_TYPES):
        return True

    venue_l = (venue or "").lower()
    if any(p.lower() in venue_l for p in _BAD_VENUE_PATTERNS):
        return True

    title_l = (title or "").lower()
    if any(p.lower() in title_l for p in _BAD_TITLE_PATTERNS):
        return True

    # Very short news-like Chinese snippets with no scholarly metadata are
    # usually not useful as paper-search results.
    has_metadata = bool(venue or publication_type or len(abstract or "") >= 80)
    if not has_metadata and re.search(r"[\u4e00-\u9fff]", title or "") and len(title or "") < 18:
        return True
    return False


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "big",
    "by",
    "for",
    "from",
    "in",
    "large",
    "method",
    "methods",
    "of",
    "on",
    "paper",
    "papers",
    "small",
    "study",
    "the",
    "to",
    "using",
    "with",
}


def _query_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _QUERY_STOPWORDS]
