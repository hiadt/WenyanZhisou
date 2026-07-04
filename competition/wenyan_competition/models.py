from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

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
            docs = [p.text() for p in papers]
            qv = self.model.encode([query], normalize_embeddings=True, batch_size=1)[0]
            dv = self.model.encode(
                docs,
                normalize_embeddings=True,
                batch_size=self.config.embedding_batch_size,
            )
            return (dv @ qv).astype(float).tolist()
        return [_sparse_cosine(query, p.text()) for p in papers]


class CrossEncoderReranker:
    """Cross-encoder reranker with heuristic fallback."""

    def __init__(self, config: SmallModelConfig, force_fallback: bool = False):
        self.config = config
        self.model = None
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
            pairs = [(query, p.title + "\n" + p.abstract[:1200]) for p in papers]
            raw = self.model.predict(pairs, batch_size=self.config.reranker_batch_size)
            return _minmax([float(x) for x in raw])
        return [_pair_heuristic(query, p) for p in papers]


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


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
