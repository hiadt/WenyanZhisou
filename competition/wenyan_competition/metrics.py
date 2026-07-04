from __future__ import annotations

from typing import Dict, Iterable, List, Set


def precision_at(pred: List[str], gold: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(pred[:k]) & gold) / k


def recall_at(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(pred[:k]) & gold) / len(gold)


def f1_at(pred: List[str], gold: Set[str], k: int) -> float:
    p = precision_at(pred, gold, k)
    r = recall_at(pred, gold, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}

