"""
Score recorded agent traces (eval/results/runs.jsonl) against eval/cases.json.

    python -m eval.score

Writes eval/results/summary.json and eval/results/report.md.

Metrics per run
  - tool selection: were all expected analysis tools called (recall), and were any
    tools called that the case does not allow (precision)?
  - argument accuracy: right stratification variable, required Cox covariates, ML cutoff
  - unavailable-variable handling: does the answer say the data cannot answer it?
  - out-of-scope handling: no analysis run, and the answer declines
  - numeric traceability: decimals quoted in the answer that cannot be traced to any
    tool output (candidate hallucinations, listed for manual review)
  - latency, LLM steps, tokens; consistency of the tool set across repeated runs
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CASES = HERE / "cases.json"
RESULTS = HERE / "results"

ALWAYS_ALLOWED = {"describe_dataset"}  # the system prompt tells the agent to start here

UNAVAILABLE_RE = re.compile(
    r"not (available|included|present|recorded|captured|collected|contain)"
    r"|does(n't| not) (contain|include|have)|no (data|information|variable|column)"
    r"|unavailable|isn't (available|in)|is not (available|in)|cannot (be )?(answer|assess|analy[sz]e|evaluate)"
    r"|can't (answer|assess|analy[sz]e|evaluate)|lack|absent|too few|insufficient|not enough",
    re.I,
)
DECLINE_RE = re.compile(
    r"\b(I can only|I'm (designed|specialized|focused)|I am (designed|specialized|focused)"
    r"|outside (of )?(my|the) (scope|expertise|dataset)|beyond (my|the) scope|not able to"
    r"|cannot (provide|help|answer|analy[sz]e|access)|can't (provide|help|answer|analy[sz]e|access)"
    r"|don't have access|do not have access|consult (a|her|your)|not (a substitute|medical advice)"
    r"|only (have access|analy[sz]e|work with))",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d+)(%?)")
THRESHOLDS = {0.05, 0.01, 0.001, 0.0001, 0.5, 1.0}


# ── helpers ──────────────────────────────────────────────────────────────────

def _flatten_numbers(obj, out):
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    return out


def _traceable(value: float, decimals: int, is_pct: bool, sources: list[float]) -> bool:
    tol = 10 ** -decimals * 0.5 + 1e-9
    for s in sources:
        candidates = [s, s * 100] if not is_pct else [s * 100, s]
        if any(abs(value - c) <= tol for c in candidates):
            return True
    return False


def untraceable_numbers(response: str, tool_outputs: list, question: str) -> list[str]:
    sources = []
    for out in tool_outputs:
        _flatten_numbers(out, sources)
    question_numbers = {m.group(1) for m in NUMBER_RE.finditer(question)}
    flagged = []
    for m in NUMBER_RE.finditer(response):
        text, is_pct = m.group(1), bool(m.group(2))
        value = float(text)
        if text in question_numbers or value in THRESHOLDS:
            continue
        before = response[max(0, m.start() - 3):m.start()]
        if re.search(r"[<>≤≥]\s*$", before):  # "p < 0.001" style thresholds
            continue
        decimals = len(text.split(".")[1])
        if not _traceable(value, decimals, is_pct, sources):
            context = response[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
            flagged.append(f"{text}{m.group(2)} … {context}")
    return flagged


def check_args(expected_args: dict, calls: list[dict]) -> list[str]:
    """Return a list of argument problems (empty = all expected args satisfied)."""
    problems = []
    for tool, spec in expected_args.items():
        tool_calls = [c for c in calls if c["name"] == tool]
        if not tool_calls:
            continue  # reported as missing_tool instead
        ok = False
        for c in tool_calls:
            a = c["args"]
            if "stratify_by" in spec and a.get("stratify_by") != spec["stratify_by"]:
                continue
            if "features_include" in spec and not set(spec["features_include"]) <= set(a.get("features", [])):
                continue
            if "cutoff_months" in spec and a.get("cutoff_months", 60) != spec["cutoff_months"]:
                continue
            ok = True
        if not ok:
            problems.append(f"{tool}: expected {spec}, got {[c['args'] for c in tool_calls]}")
    return problems


def score_run(case: dict, run: dict) -> dict:
    calls = run["tool_calls"]
    called = [c["name"] for c in calls]
    called_set = set(called)
    expected = set(case["expected_tools"])
    allowed = expected | set(case.get("optional_tools", []))
    if case["category"] != "out_of_scope":
        allowed |= ALWAYS_ALLOWED

    labels = []
    if run["error"]:
        labels.append("recursion_limit" if "Recursion" in run["error"] else "exception")
    missing = sorted(expected - called_set)
    unexpected = sorted(called_set - allowed)
    if missing:
        labels.append("missing_tool")
    if unexpected:
        labels.append("unnecessary_tool" if case["category"] == "out_of_scope" else "unexpected_tool")
    arg_problems = check_args(case.get("expected_args", {}), calls)
    if arg_problems:
        labels.append("wrong_args")

    tool_errors = [c for c in calls if isinstance(c.get("result"), dict) and "error" in c["result"]]
    response = run["response"]
    if case["category"] == "unavailable_variable" and not UNAVAILABLE_RE.search(response):
        labels.append("unavailability_not_stated")
    if case["category"] == "out_of_scope" and not DECLINE_RE.search(response):
        labels.append("did_not_decline")

    untraceable = untraceable_numbers(
        response, [c.get("result") for c in calls if c.get("result") is not None], case["question"]
    )

    recall = len(expected & called_set) / len(expected) if expected else 1.0
    precision = len(called_set & allowed) / len(called_set) if called_set else 1.0
    return {
        "case_id": case["id"],
        "category": case["category"],
        "run": run["run"],
        "success": not labels,
        "labels": labels,
        "tool_sequence": called,
        "tool_recall": recall,
        "tool_precision": precision,
        "missing_tools": missing,
        "unexpected_tools": unexpected,
        "arg_problems": arg_problems,
        "tool_errors": [{"tool": c["name"], "error": c["result"]["error"]} for c in tool_errors],
        "untraceable_numbers": untraceable,
        "latency_s": run["latency_s"],
        "llm_steps": run["llm_steps"],
        "tokens": run["input_tokens"] + run["output_tokens"],
    }


def _pct(x):
    return f"{100 * x:.0f}%"


def _quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cases = {c["id"]: c for c in json.loads(CASES.read_text())}
    runs = [json.loads(l) for l in (RESULTS / "runs.jsonl").read_text().splitlines() if l.strip()]
    scored = [score_run(cases[r["case_id"]], r) for r in runs if r["case_id"] in cases]

    by_case = defaultdict(list)
    for s in scored:
        by_case[s["case_id"]].append(s)
    by_cat = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s)

    def consistency(rs):
        sets = Counter(tuple(sorted(set(r["tool_sequence"]))) for r in rs)
        return sets.most_common(1)[0][1] / len(rs)

    label_counts = Counter(l for s in scored for l in s["labels"])
    n_untraceable_runs = sum(1 for s in scored if s["untraceable_numbers"])
    lat = [s["latency_s"] for s in scored]

    summary = {
        "n_cases": len(by_case),
        "n_runs": len(scored),
        "task_success_rate": sum(s["success"] for s in scored) / len(scored),
        "tool_recall": statistics.mean(s["tool_recall"] for s in scored),
        "tool_precision": statistics.mean(s["tool_precision"] for s in scored),
        "arg_accuracy": 1 - sum("wrong_args" in s["labels"] for s in scored) / len(scored),
        "runs_with_untraceable_numbers": n_untraceable_runs / len(scored),
        "tool_set_consistency": statistics.mean(consistency(rs) for rs in by_case.values()),
        "latency_p50_s": _quantile(lat, 0.5),
        "latency_p95_s": _quantile(lat, 0.95),
        "mean_llm_steps": statistics.mean(s["llm_steps"] for s in scored),
        "mean_tokens": statistics.mean(s["tokens"] for s in scored),
        "failure_labels": dict(label_counts),
        "by_category": {
            cat: {"runs": len(rs), "success_rate": sum(r["success"] for r in rs) / len(rs)}
            for cat, rs in sorted(by_cat.items())
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "scored_runs.json").write_text(json.dumps(scored, indent=2))

    lines = [
        "# Agent evaluation report",
        "",
        f"{summary['n_cases']} cases × {summary['n_runs'] // max(1, summary['n_cases'])} runs = {summary['n_runs']} agent runs.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success (no failure label) | {_pct(summary['task_success_rate'])} |",
        f"| Tool recall (expected tools called) | {_pct(summary['tool_recall'])} |",
        f"| Tool precision (calls that were allowed) | {_pct(summary['tool_precision'])} |",
        f"| Argument accuracy | {_pct(summary['arg_accuracy'])} |",
        f"| Runs quoting numbers not traceable to tool output | {_pct(summary['runs_with_untraceable_numbers'])} |",
        f"| Tool-set consistency across repeated runs | {_pct(summary['tool_set_consistency'])} |",
        f"| Latency p50 / p95 | {summary['latency_p50_s']:.1f} s / {summary['latency_p95_s']:.1f} s |",
        f"| Mean LLM steps / tokens per run | {summary['mean_llm_steps']:.1f} / {summary['mean_tokens']:.0f} |",
        "",
        "## By category",
        "",
        "| Category | Runs | Success |",
        "|---|---|---|",
    ]
    for cat, v in summary["by_category"].items():
        lines.append(f"| {cat} | {v['runs']} | {_pct(v['success_rate'])} |")

    lines += ["", "## Failure modes", "", "| Label | Runs |", "|---|---|"]
    for label, n in label_counts.most_common():
        lines.append(f"| {label} | {n} |")

    lines += ["", "## Per case", "", "| Case | Success | Consistency | Tool sequences seen | Labels |", "|---|---|---|---|---|"]
    for cid, rs in by_case.items():
        seqs = Counter(" → ".join(r["tool_sequence"]) or "(none)" for r in rs)
        seq_txt = "; ".join(f"{k} ×{v}" for k, v in seqs.most_common())
        labs = Counter(l for r in rs for l in r["labels"])
        lab_txt = ", ".join(f"{k} ×{v}" for k, v in labs.most_common()) or ""
        lines.append(
            f"| {cid} | {sum(r['success'] for r in rs)}/{len(rs)} | {_pct(consistency(rs))} | {seq_txt} | {lab_txt} |"
        )

    review = [(s["case_id"], s["run"], n) for s in scored for n in s["untraceable_numbers"]]
    if review:
        lines += ["", "## Numbers to review by hand", "",
                  "Decimals in the answer that match no tool output. Some are legitimate "
                  "(derived values such as 1 − HR, or literature figures); others are hallucinations.", ""]
        for cid, run, n in review[:60]:
            lines.append(f"- `{cid}` run {run}: {n}")

    arg_rows = [(s["case_id"], s["run"], p) for s in scored for p in s["arg_problems"]]
    if arg_rows:
        lines += ["", "## Argument problems", ""]
        for cid, run, p in arg_rows:
            lines.append(f"- `{cid}` run {run}: {p}")

    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nFull report: {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
