from __future__ import annotations

import argparse
import json
from pathlib import Path

from wenyan_competition.agent import AcademicSearchAgent
from wenyan_competition.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="runs/single_result.json")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM planner/verifier for smoke tests.")
    parser.add_argument("--fallback_models", action="store_true", help="Use sparse/heuristic fallback instead of loading model weights.")
    args = parser.parse_args()

    config = load_config(args.config)
    agent = AcademicSearchAgent(config, use_llm=not args.no_llm, force_fallback_models=args.fallback_models)
    result = agent.search(args.query, top_k=args.top_k)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(result.summary)


if __name__ == "__main__":
    main()

