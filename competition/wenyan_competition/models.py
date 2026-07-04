from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .config import SmallModelConfig
from .schema import Paper


class EmbeddingModel:
    """Embedding scorer with graceful fallback.

    Preferred path: sentence-transformers model, e.g. BAAI/bge-small-en-v1.5.
    Fallback path: sparse cosine over token counts, so smoke tests can run
    without downloading weights.
    """

    def __init__(self, config: SmallModelConfig, force_fallback: bool = False):
        self.config = config
        self.model = None
        self._query_cache: Dict[str, object] = {}
        self._doc_cache: Dict[str, object] = {}
        self._score_cache: Dict[Tuple[str, str], float] = {}
        if not force_fallback:
            try:
                from sentence_transformers import SentenceTransformer

                device = None if config.device == "auto" else config.device
                self.model = SentenceTransformer(config.embedding_model, device=device)
            except Exception:
                self.model = None

    def score(self, query: str, papers: Sequence[Paper]) -> List[float]:
        if not papers:
            return []
        if self.model is not None:
            if query not in self._query_cache:
                self._query_cache[query] = self.model.encode([query], normalize_embeddings=True, batch_size=1)[0]
            qv = self._query_cache[query]
            doc_keys = [_paper_text_key(p) for p in papers]
            missing = [(key, p.text()) for key, p in zip(doc_keys, papers) if key not in self._doc_cache]
            if missing:
                encoded = self.model.encode(
                    [text for _, text in missing],
                    normalize_embeddings=True,
                    batch_size=self.config.embedding_batch_size,
                )
                for (key, _), vector in zip(missing, encoded):
                    self._doc_cache[key] = vector
            return [float(self._doc_cache[key] @ qv) for key in doc_keys]

        scores = []
        for p in papers:
            key = (query, _paper_text_key(p))
            if key not in self._score_cache:
                self._score_cache[key] = _sparse_cosine(query, p.text())
            scores.append(self._score_cache[key])
        return scores


class CrossEncoderReranker:
    """Cross-encoder reranker with heuristic fallback."""

    def __init__(self, config: SmallModelConfig, force_fallback: bool = False):
        self.config = config
        self.model = None
        self._score_cache: Dict[Tuple[str, str], float] = {}
        if not force_fallback:
            try:
                from sentence_transformers import CrossEncoder

                device = None if config.device == "auto" else config.device
                self.model = CrossEncoder(config.reranker_model, device=device)
            except Exception:
                self.model = None

    def score(self, query: str, papers: Sequence[Paper]) -> List[float]:
        if not papers:
            return []
        if self.model is not None:
            keys = [(query, _paper_rerank_key(p)) for p in papers]
            missing = [(key, p) for key, p in zip(keys, papers) if key not in self._score_cache]
            if missing:
                pairs = [(query, p.title + "\n" + p.abstract[:1200]) for _, p in missing]
                raw = self.model.predict(pairs, batch_size=self.config.reranker_batch_size)
                for (key, _), value in zip(missing, raw):
                    self._score_cache[key] = float(value)
            return _minmax([self._score_cache[key] for key in keys])

        scores = []
        for p in papers:
            key = (query, _paper_rerank_key(p))
            if key not in self._score_cache:
                self._score_cache[key] = _pair_heuristic(query, p)
            scores.append(self._score_cache[key])
        return scores


def bm25_like_scores(query: str, papers: Sequence[Paper]) -> List[float]:
    q = _tokens(query)
    if not q or not papers:
        return [0.0 for _ in papers]
    docs = [_tokens(p.text()) for p in papers]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    scores = []
    k1 = 1.2
    b = 0.75
    for d in docs:
        tf = Counter(d)
        dl = len(d)
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(1.0, avgdl))
            score += idf * (tf[term] * (k1 + 1) / denom)
        scores.append(score)
    return _minmax(scores)


def _sparse_cosine(a: str, b: str) -> float:
    ca = Counter(_tokens(a))
    cb = Counter(_tokens(b))
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / max(1e-9, na * nb)


def _pair_heuristic(query: str, paper: Paper) -> float:
    q = set(_tokens(query))
    title = set(_tokens(paper.title))
    body = set(_tokens(paper.text()))
    if not q:
        return 0.0
    title_hit = len(q & title) / len(q)
    body_hit = len(q & body) / len(q)
    phrase_bonus = 0.15 if _important_phrase_hit(query, paper.text()) else 0.0
    return min(1.0, 0.55 * title_hit + 0.35 * body_hit + phrase_bonus)


def _important_phrase_hit(query: str, text: str) -> bool:
    q = query.lower()
    t = text.lower()
    phrases = [
        "large language model",
        "retrieval augmented generation",
        "graph neural network",
        "vulnerability detection",
        "hallucination detection",
        "semantic similarity",
        "intelligent vehicle",
        "autonomous vehicle",
        "connected automated vehicle",
        "functional safety",
        "system safety",
        "safety strategy",
        "cybersecurity",
        "brake force distribution",
        "heavy truck",
        "commercial vehicle",
        "smart cockpit",
        "intelligent cockpit",
        "automotive cockpit",
        "vehicle cabin",
        "in-vehicle infotainment",
        "human machine interaction",
        "adaptive control",
        "automatic adjustment",
        "personalization",
        "thermal comfort",
        "climate control",
    ]
    return any(p in q and p in t for p in phrases)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())


def _paper_text_key(paper: Paper) -> str:
    return _stable_key("|".join([paper.key(), paper.title or "", paper.abstract[:500] or "", paper.full_text[:500] or ""]))


def _paper_rerank_key(paper: Paper) -> str:
    return _stable_key("|".join([paper.key(), paper.title or "", paper.abstract[:1200] or ""]))


def _stable_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=12).hexdigest()


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
