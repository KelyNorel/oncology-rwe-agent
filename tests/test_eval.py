"""Tests for the evaluation harness itself (scoring rules and tracing), no API calls."""

import json

from langchain_core.messages import AIMessage

import src.agent as agent
from eval import run_eval, score
from tests.test_agent import ScriptedLLM, tool_call

CASES = {c["id"]: c for c in json.loads(score.CASES.read_text())}


def _run(calls, response, error=None):
    return {"run": 0, "tool_calls": calls, "response": response, "error": error,
            "latency_s": 1.0, "llm_steps": 2, "input_tokens": 10, "output_tokens": 5}


def test_cases_file_is_valid():
    tools = {"describe_dataset", "kaplan_meier_analysis", "cox_model", "ml_prediction"}
    for c in CASES.values():
        assert set(c["expected_tools"]) <= tools
        assert set(c.get("optional_tools", [])) <= tools
        assert set(c.get("expected_args", {})) <= set(c["expected_tools"])


def test_correct_single_tool_run_succeeds():
    calls = [{"name": "kaplan_meier_analysis", "args": {"stratify_by": "subtype"},
              "result": {"logrank_p": 0.0003}}]
    s = score.score_run(CASES["km_subtype"], _run(calls, "Log-rank p = 0.0003."))
    assert s["success"] and s["labels"] == [] and s["untraceable_numbers"] == []


def test_wrong_stratification_variable_is_flagged():
    calls = [{"name": "kaplan_meier_analysis", "args": {"stratify_by": "her2"}, "result": {}}]
    s = score.score_run(CASES["km_er"], _run(calls, "..."))
    assert "wrong_args" in s["labels"]


def test_missing_covariate_is_flagged():
    calls = [{"name": "cox_model", "args": {"features": ["lymph_nodes", "age"]}, "result": {}}]
    s = score.score_run(CASES["cox_nodes_adjusted"], _run(calls, "..."))
    assert "wrong_args" in s["labels"]


def test_describe_dataset_is_always_allowed_except_out_of_scope():
    calls = [{"name": "describe_dataset", "args": {}, "result": {}},
             {"name": "ml_prediction", "args": {"cutoff_months": 60}, "result": {}}]
    assert score.score_run(CASES["ml_5yr"], _run(calls, "AUC ..."))["success"]

    calls = [{"name": "describe_dataset", "args": {}, "result": {}}]
    s = score.score_run(CASES["oos_poem"], _run(calls, "I can only analyze the METABRIC data."))
    assert "unnecessary_tool" in s["labels"]


def test_out_of_scope_must_decline():
    assert score.score_run(CASES["oos_poem"], _run([], "I can only help with the METABRIC cohort."))["success"]
    s = score.score_run(CASES["oos_poem"], _run([], "Waves roll in, blue and deep..."))
    assert "did_not_decline" in s["labels"]


def test_unavailable_variable_must_be_stated():
    ok = score.score_run(CASES["unavail_race"], _run([], "Race is not available in this dataset."))
    assert ok["success"]
    bad = score.score_run(CASES["unavail_race"], _run([], "White patients survive longer."))
    assert "unavailability_not_stated" in bad["labels"]


def test_untraceable_numbers():
    outputs = [{"c_index": 0.762, "hazard_ratios": {"exp(coef)": {"grade": 1.45}}, "outcome_rate": 0.231}]
    resp = "C-index 0.76, HR 1.45 for grade, 23.1% died, p < 0.05, and HR 2.10 for age."
    flagged = score.untraceable_numbers(resp, outputs, "question")
    assert len(flagged) == 1 and flagged[0].startswith("2.10")


def test_trace_run_records_tools_and_tokens(monkeypatch, cohort_csv):
    first = AIMessage(content="", tool_calls=[tool_call("ml_prediction", {"cutoff_months": 60}, "c1")],
                      usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
    final = AIMessage(content="Best AUC reported.",
                      usage_metadata={"input_tokens": 300, "output_tokens": 50, "total_tokens": 350})
    monkeypatch.setattr(agent, "llm_with_tools", ScriptedLLM([first, final]))

    t = run_eval.trace_run("Predict 5-year mortality", recursion_limit=12)
    assert t["error"] is None
    assert [c["name"] for c in t["tool_calls"]] == ["ml_prediction"]
    assert "auc_scores" in t["tool_calls"][0]["result"]
    assert (t["input_tokens"], t["output_tokens"], t["llm_steps"]) == (400, 70, 2)
    json.dumps(t)
