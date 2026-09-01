"""InspectAI solver that connects AstaBench to a local Wenyan service.

The official AstaBench runtime requires Python 3.11+, while the validated
WenyanZhiSou model environment uses Python 3.10.  Keeping the environments
separate avoids changing CUDA/model dependencies in the competition pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.request import Request, urlopen

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or not isinstance(body.get("output"), dict):
        raise RuntimeError("Wenyan ASTA service returned an invalid response")
    return body


@solver
def wenyan_paper_finder(
    backend_url: str = "http://127.0.0.1:8765/solve",
    top_k: int = 100,
    timeout: int = 240,
) -> Solver:
    """Return official PaperFindingBench JSON produced by WenyanZhiSou.

    Start ``asta_solver_service.py`` in the validated project environment
    before running Inspect.  ``--max-samples 1`` is recommended because the
    backend intentionally serializes access to model instances and counters.
    """

    requested_top_k = max(1, min(int(top_k), 250))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        query_id = str(state.sample_id or "")
        query = str((state.metadata or {}).get("raw_query") or "").strip()
        if not query:
            raise RuntimeError("AstaBench sample metadata is missing raw_query")
        payload = await asyncio.to_thread(
            _post_json,
            backend_url,
            {"query_id": query_id, "query": query, "top_k": requested_top_k},
            timeout,
        )
        state.output = ModelOutput.from_content(
            "wenyan-zhisou",
            json.dumps(payload, ensure_ascii=False),
        )
        return state

    return solve

