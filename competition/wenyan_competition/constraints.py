from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import Paper, QueryPlan


@dataclass
class ConstraintSpec:
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    venues: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)

    def active_dimensions(self) -> List[Tuple[str, List[str]]]:
        dimensions: List[Tuple[str, List[str]]] = []
        if self.min_year is not None or self.max_year is not None:
            lo = str(self.min_year) if self.min_year is not None else "-"
            hi = str(self.max_year) if self.max_year is not None else "-"
            dimensions.append(("year", [f"{lo}..{hi}"]))
        for name, values in [
            ("venue", self.venues),
            ("publication_type", self.publication_types),
            ("method", self.methods),
            ("dataset", self.datasets),
            ("topic", self.entities),
        ]:
            cleaned = _unique(values)
            if cleaned:
                dimensions.append((name, cleaned))
        return dimensions


def parse_constraint_spec(query: str, plan: QueryPlan) -> ConstraintSpec:
    flat_constraints = list(_flatten_constraints(plan.constraints))
    joined = " ".join([query, *flat_constraints])
    min_year, max_year = _year_bounds(joined, plan.constraints)
    return ConstraintSpec(
        min_year=min_year,
        max_year=max_year,
        venues=_values_for_keys(plan.constraints, ("venue", "conference", "journal", "会议", "期刊")),
        publication_types=_values_for_keys(
            plan.constraints,
            ("publication_type", "paper_type", "document_type", "文献类型", "论文类型"),
        ),
        methods=_clean_plan_terms(plan.methods),
        datasets=_clean_plan_terms(plan.datasets),
        entities=_clean_plan_terms(plan.entities),
        negative_terms=_unique(plan.negative_terms),
    )


def apply_constraint_policy(
    query: str,
    plan: QueryPlan,
    papers: List[Paper],
    top_k: int,
    weight: float = 0.04,
    hard_filter_year: bool = True,
) -> List[Paper]:
    """Annotate constraints and conservatively adjust an existing ranking.

    The incoming order remains the dominant signal. Constraint evidence can
    move a paper only a few positions. An explicit year mismatch is removed
    only when enough non-conflicting candidates remain to satisfy Top K.
    """

    if not papers:
        return []
    spec = parse_constraint_spec(query, plan)
    dimensions = spec.active_dimensions()
    if not dimensions and not spec.negative_terms:
        return papers

    evaluations = []
    for paper in papers:
        matched, missed, unknown, year_violation = _evaluate_paper(spec, paper)
        known = len(matched) + len(missed)
        paper.constraint_score = len(matched) / max(1, known)
        paper.constraint_coverage = len(matched) / max(1, len(dimensions))
        paper.constraint_matches = matched
        paper.constraint_misses = missed
        paper.constraint_unknown = unknown
        evaluations.append((paper, year_violation))

    eligible = [paper for paper, violation in evaluations if not violation]
    if hard_filter_year and len(eligible) >= max(1, top_k):
        ranked_pool = eligible
    else:
        ranked_pool = [paper for paper, _ in evaluations]

    base_size = max(1, len(ranked_pool))
    base_rank = {id(paper): idx for idx, paper in enumerate(ranked_pool)}

    def score(paper: Paper) -> float:
        rank_quality = 1.0 - base_rank[id(paper)] / base_size
        negative_penalty = 0.03 if any(x.startswith("negative:") for x in paper.constraint_misses) else 0.0
        return rank_quality + max(0.0, weight) * paper.constraint_score - negative_penalty

    return sorted(ranked_pool, key=score, reverse=True)


def build_constraint_coverage(query: str, plan: QueryPlan, papers: List[Paper]) -> Dict[str, Any]:
    spec = parse_constraint_spec(query, plan)
    rows = []
    evidence = [_paper_constraint_evidence(spec, paper) for paper in papers]
    for name, requirements in spec.active_dimensions():
        prefix = name + ":"
        matched = sum(any(item.startswith(prefix) for item in item[0]) for item in evidence)
        missed = sum(any(item.startswith(prefix) for item in item[1]) for item in evidence)
        known = matched + missed
        rows.append(
            {
                "dimension": name,
                "requirements": requirements,
                "matched_papers": matched,
                "known_papers": known,
                "total_papers": len(papers),
                "coverage": matched / max(1, len(papers)),
                "status": "covered" if matched else ("missing" if known else "unknown"),
            }
        )
    return {
        "dimensions": rows,
        "covered_dimensions": sum(row["status"] == "covered" for row in rows),
        "total_dimensions": len(rows),
        "overall_coverage": (
            sum(row["coverage"] for row in rows) / len(rows) if rows else 1.0
        ),
    }


def constraint_gap_queries(
    query: str,
    plan: QueryPlan,
    papers: List[Paper],
    max_queries: int = 2,
    min_coverage: float = 0.25,
) -> List[str]:
    """Build deterministic second-round queries for uncovered dimensions."""

    if max_queries <= 0:
        return []
    coverage = build_constraint_coverage(query, plan, papers)
    rows = [row for row in coverage["dimensions"] if row["coverage"] < min_coverage]
    priority = {"dataset": 0, "method": 1, "venue": 2, "year": 3, "publication_type": 4, "topic": 5}
    rows.sort(key=lambda row: (priority.get(row["dimension"], 9), row["coverage"]))
    core_terms = _unique([*plan.entities[:3], plan.intent or query])
    core = " ".join(core_terms) or query
    queries = []
    for row in rows:
        requirements = [str(x).replace("..", " ") for x in row["requirements"][:3]]
        focused = " ".join(_unique([core, *requirements])).strip()
        if focused and _normalize_text(focused) != _normalize_text(query):
            queries.append(focused)
        if len(queries) >= max_queries:
            break
    return _unique(queries)


def _paper_constraint_evidence(spec: ConstraintSpec, paper: Paper) -> Tuple[List[str], List[str], List[str], bool]:
    if paper.constraint_matches or paper.constraint_misses or paper.constraint_unknown:
        year_violation = any(item.startswith("year:") for item in paper.constraint_misses)
        return paper.constraint_matches, paper.constraint_misses, paper.constraint_unknown, year_violation
    return _evaluate_paper(spec, paper)


def _evaluate_paper(spec: ConstraintSpec, paper: Paper) -> Tuple[List[str], List[str], List[str], bool]:
    matched: List[str] = []
    missed: List[str] = []
    unknown: List[str] = []
    year_violation = False
    text = " ".join([paper.title, paper.abstract, paper.full_text[:1500], paper.venue, paper.publication_type])

    if spec.min_year is not None or spec.max_year is not None:
        if paper.year is None:
            unknown.append("year:metadata unavailable")
        else:
            in_range = (spec.min_year is None or paper.year >= spec.min_year) and (
                spec.max_year is None or paper.year <= spec.max_year
            )
            if in_range:
                matched.append(f"year:{paper.year}")
            else:
                missed.append(f"year:{paper.year}")
                year_violation = True

    _match_dimension("venue", spec.venues, paper.venue, matched, missed, unknown)
    _match_dimension("publication_type", spec.publication_types, paper.publication_type, matched, missed, unknown)
    _match_dimension("method", spec.methods, text, matched, missed, unknown)
    _match_dimension("dataset", spec.datasets, text, matched, missed, unknown)
    _match_dimension("topic", spec.entities, text, matched, missed, unknown)

    for term in spec.negative_terms:
        if _term_matches(term, text):
            missed.append(f"negative:{term}")
    return matched, missed, unknown, year_violation


def _match_dimension(
    name: str,
    terms: List[str],
    text: str,
    matched: List[str],
    missed: List[str],
    unknown: List[str],
) -> None:
    if not terms:
        return
    if not (text or "").strip():
        unknown.append(f"{name}:metadata unavailable")
        return
    hits = [term for term in terms if _term_matches(term, text)]
    if hits:
        matched.append(f"{name}:{', '.join(hits[:3])}")
    else:
        missed.append(f"{name}:{', '.join(terms[:3])}")


def _term_matches(term: str, text: str) -> bool:
    term_n = _normalize_text(term)
    text_n = _normalize_text(text)
    if not term_n or not text_n:
        return False
    if term_n in text_n:
        return True
    term_tokens = [x for x in term_n.split() if len(x) > 1]
    if not term_tokens:
        return False
    hits = sum(token in text_n for token in term_tokens)
    return hits >= max(1, (len(term_tokens) + 1) // 2)


def _year_bounds(text: str, constraints: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    min_year = _numeric_constraint(constraints, ("min_year", "start_year", "year_from", "from_year"))
    max_year = _numeric_constraint(constraints, ("max_year", "end_year", "year_to", "to_year"))
    normalized = (text or "").lower()
    range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to|through|至|到)\s*((?:19|20)\d{2})\b", normalized)
    if range_match:
        lo, hi = sorted((int(range_match.group(1)), int(range_match.group(2))))
        min_year = lo if min_year is None else max(min_year, lo)
        max_year = hi if max_year is None else min(max_year, hi)
    lower = re.search(r"(?:since|after|from|>=|not before|自|不早于)\s*((?:19|20)\d{2})", normalized)
    lower_cn = re.search(r"((?:19|20)\d{2})\s*年?(?:以来|以后|之后|后)", normalized)
    upper = re.search(r"(?:before|until|through|<=|not after|截至|不晚于)\s*((?:19|20)\d{2})", normalized)
    upper_cn = re.search(r"((?:19|20)\d{2})\s*年?(?:以前|之前|前)", normalized)
    if lower or lower_cn:
        value = int((lower or lower_cn).group(1))
        min_year = value if min_year is None else max(min_year, value)
    if upper or upper_cn:
        value = int((upper or upper_cn).group(1))
        max_year = value if max_year is None else min(max_year, value)
    recent = re.search(r"(?:last|past|最近|近)\s*(\d{1,2})\s*(?:years?|年)", normalized)
    if recent:
        value = datetime.now().year - int(recent.group(1)) + 1
        min_year = value if min_year is None else max(min_year, value)
    return min_year, max_year


def _numeric_constraint(data: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    lowered = {str(k).lower(): v for k, v in (data or {}).items()}
    for key in keys:
        value = lowered.get(key)
        match = re.search(r"\b((?:19|20)\d{2})\b", str(value or ""))
        if match:
            return int(match.group(1))
    return None


def _values_for_keys(data: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    values: List[str] = []
    key_terms = tuple(x.lower() for x in keys)
    for key, value in (data or {}).items():
        if any(term in str(key).lower() for term in key_terms):
            values.extend(_as_values(value))
    return _unique(values)


def _flatten_constraints(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_constraints(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_constraints(item)
    elif value not in (None, "", False):
        yield str(value)


def _as_values(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, dict):
        return [str(x) for x in value.values() if str(x).strip()]
    return [str(value)] if str(value or "").strip() else []


def _clean_plan_terms(values: List[str]) -> List[str]:
    generic = {"method", "methods", "dataset", "datasets", "paper", "papers", "study", "research"}
    return [value for value in _unique(values) if _normalize_text(value) not in generic][:10]


def _normalize_text(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (value or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
