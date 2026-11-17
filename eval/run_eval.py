"""
Run the oncology RWE agent on the evaluation cases and record full traces.

Usage (from the repo root, with ANTHROPIC_API_KEY in .env and the METABRIC data in place):

    python -m eval.run_eval --runs 3

Each run is appended to eval/results/runs.jsonl as soon as it finishes, so an
interrupted run can be resumed with the same command (finished runs are skipped).
Score afterwards with:  python -m eval.score
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

CASES = Path(__file__).parent / "cases.json"
RESULTS = Path(__file__).parent / "results"


def _parse_tool_output(content):
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(content)[:2000]}
    if isinstance(data, dict):
        data.pop("plot_base64", None)
    return data


def trace_run(question: str, recursion_limit: int) -> dict:
    """Run the agent once and return a JSON-serializable trace."""
    from src.agent import app, HumanMessage

    t0 = time.perf_counter()
    try:
        state = app.invoke(
            {"messages": [HumanMessage(content=question)], "plots": []},
            config={"recursion_limit": recursion_limit},
        )
        error = None
    except Exception as exc:  # recorded as a failure mode, not raised
        state, error = None, f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    latency = time.perf_counter() - t0

    tool_calls, tool_results, llm_steps = [], {}, 0
    input_tokens = output_tokens = 0
    response = ""
    if state is not None:
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                llm_steps += 1
                usage = getattr(msg, "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                for tc in msg.tool_calls:
                    tool_calls.append({"id": tc["id"], "name": tc["name"], "args": tc["args"]})
            elif isinstance(msg, ToolMessage):
                tool_results[msg.tool_call_id] = _parse_tool_output(msg.content)
        final = state["messages"][-1].content
        if isinstance(final, list):
            final = " ".join(b.get("text", "") for b in final if isinstance(b, dict))
        response = str(final)

    for tc in tool_calls:
        tc["result"] = tool_results.get(tc["id"])

    return {
        "latency_s": round(latency, 2),
        "llm_steps": llm_steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "response": response,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="repetitions per case")
    parser.add_argument("--cases", nargs="*", help="only run these case ids")
    parser.add_argument("--recursion-limit", type=int, default=25)  # LangGraph default, same as the app
    parser.add_argument("--out", default=str(RESULTS / "runs.jsonl"))
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())
    if args.cases:
        cases = [c for c in cases if c["id"] in set(args.cases)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["case_id"], r["run"]))

    todo = [(c, i) for c in cases for i in range(args.runs) if (c["id"], i) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already recorded)")

    for n, (case, i) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {case['id']} run {i}", flush=True)
        record = {"case_id": case["id"], "run": i, **trace_run(case["question"], args.recursion_limit)}
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"Done. Traces in {out}. Next: python -m eval.score")


if __name__ == "__main__":
    main()
