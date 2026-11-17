"""
Tests for the LangGraph control flow, using a scripted fake LLM.

These check the orchestration (routing to tools, looping back, stopping), not the
quality of Claude's reasoning; that is what eval/ measures.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import src.agent as agent


class ScriptedLLM:
    """Returns pre-written AIMessages in order and records what it was shown."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def scripted(monkeypatch, cohort_csv):
    def _install(responses):
        llm = ScriptedLLM(responses)
        monkeypatch.setattr(agent, "llm_with_tools", llm)
        return llm
    return _install


def test_should_continue_routes_on_tool_calls():
    with_calls = AIMessage(content="", tool_calls=[tool_call("describe_dataset", {}, "1")])
    assert agent.should_continue({"messages": [with_calls]}) == "tools"
    assert agent.should_continue({"messages": [AIMessage(content="done")]}) == "end"


def test_single_tool_then_answer(scripted):
    llm = scripted([
        AIMessage(content="", tool_calls=[tool_call("kaplan_meier_analysis", {"stratify_by": "subtype"}, "c1")]),
        AIMessage(content="Survival differs by subtype."),
    ])
    out = agent.run_agent("Compare survival by subtype")

    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["kaplan_meier_analysis"]
    assert out["response"] == "Survival differs by subtype."
    assert len(out["plots"]) == 1
    assert len(llm.calls) == 2


def test_plots_are_stripped_before_returning_to_llm(scripted):
    llm = scripted([
        AIMessage(content="", tool_calls=[tool_call("describe_dataset", {}, "c1")]),
        AIMessage(content="Here is the overview."),
    ])
    agent.run_agent("Describe the dataset")

    second_call = llm.calls[1]
    tool_results = [m for m in second_call if isinstance(m, ToolMessage)]
    assert tool_results, "the LLM should see the tool result"
    assert "plot_base64" not in json.loads(tool_results[0].content)


def test_multi_tool_sequence(scripted):
    scripted([
        AIMessage(content="", tool_calls=[tool_call("kaplan_meier_analysis", {"stratify_by": "hormone_tx"}, "c1")]),
        AIMessage(content="", tool_calls=[tool_call("cox_model", {"features": ["hormone_bin", "age", "grade"]}, "c2")]),
        AIMessage(content="Adjusted for age and grade, ..."),
    ])
    out = agent.run_agent("Is hormone therapy associated with survival? Control for confounders.")
    names = [m.name for m in out["messages"] if isinstance(m, ToolMessage)]
    assert names == ["kaplan_meier_analysis", "cox_model"]


def test_out_of_scope_question_calls_no_tools(scripted):
    scripted([AIMessage(content="I can only analyze the METABRIC breast cancer cohort.")])
    out = agent.run_agent("What is the best treatment for lung cancer?")
    assert not [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert out["plots"] == []


def test_tool_error_is_returned_to_llm_not_raised(scripted):
    llm = scripted([
        AIMessage(content="", tool_calls=[tool_call("kaplan_meier_analysis", {"stratify_by": "race"}, "c1")]),
        AIMessage(content="Race is not available in this dataset."),
    ])
    out = agent.run_agent("Survival by race?")
    result = json.loads([m for m in llm.calls[1] if isinstance(m, ToolMessage)][0].content)
    assert "error" in result
    assert out["response"].startswith("Race is not available")


def test_list_content_blocks_are_flattened(scripted):
    scripted([AIMessage(content=[{"type": "text", "text": "Part one."}, {"type": "text", "text": "Part two."}])])
    out = agent.run_agent("hello")
    assert out["response"] == "Part one. Part two."


def test_recursion_limit_stops_a_looping_agent(scripted):
    looping = [
        AIMessage(content="", tool_calls=[tool_call("describe_dataset", {}, f"c{i}")]) for i in range(50)
    ]
    scripted(looping)
    from langgraph.errors import GraphRecursionError

    with pytest.raises(GraphRecursionError):
        agent.app.invoke(
            {"messages": [HumanMessage(content="loop")], "plots": []},
            config={"recursion_limit": 6},
        )
