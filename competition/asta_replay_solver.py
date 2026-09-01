"""Replay frozen Wenyan ASTA predictions through the official Inspect scorer.

This solver is evaluation plumbing only. It reads the exact ``asta_output``
objects previously written by ``evaluate_asta_paper_finder.py`` and never
invokes retrieval, ranking, or an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver


def _load_predictions(path: str) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            output = row.get("asta_output")
            if not query_id or not isinstance(output, dict):
                raise ValueError(f"invalid prediction at line {line_number}")
            predictions[query_id] = output
    return predictions


@solver
def replay_wenyan_predictions(predictions_path: str) -> Solver:
    """Return stored official-format output for each AstaBench sample ID."""

    predictions = _load_predictions(predictions_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        query_id = str(state.sample_id or "")
        if query_id not in predictions:
            raise RuntimeError(f"no frozen prediction for sample {query_id!r}")
        state.output = ModelOutput.from_content(
            "wenyan-zhisou-replay",
            json.dumps(predictions[query_id], ensure_ascii=False),
        )
        return state

    return solve
