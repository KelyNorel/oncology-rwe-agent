"""
agent.py — LangGraph-based oncology RWE co-scientist agent
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
from langchain_core.messages import ToolMessage
import json, ast

from src.tools import (
    describe_dataset,
    kaplan_meier_analysis,
    cox_model,
    ml_prediction,
)
from src.prompts import SYSTEM_PROMPT

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1500,
)

tools = [describe_dataset, kaplan_meier_analysis, cox_model, ml_prediction]
llm_with_tools = llm.bind_tools(tools)

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plots: Annotated[list, operator.add]

# ── Nodes ─────────────────────────────────────────────────────────────────────
def strip_plots_from_messages(messages):
    """Remove plot_base64 from tool results to reduce token usage."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "plot_base64" in data:
                    data_clean = {k: v for k, v in data.items() if k != "plot_base64"}
                    msg = ToolMessage(
                        content=json.dumps(data_clean),
                        tool_call_id=msg.tool_call_id
                    )
            except (json.JSONDecodeError, TypeError):
                pass
        cleaned.append(msg)
    return cleaned

def scientist_node(state: AgentState) -> AgentState:
    """The co-scientist reasons and decides what to do next."""
    cleaned_messages = strip_plots_from_messages(list(state["messages"]))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + cleaned_messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "plots": []}

def extract_plots(state: AgentState) -> AgentState:
    """Extract base64 plots from tool results and add to state."""
    from langchain_core.messages import ToolMessage
    import json
    
    plots = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "plot_base64" in data:
                    plots.append(data["plot_base64"])
            except (json.JSONDecodeError, TypeError):
                pass
    return {"messages": [], "plots": plots}

def should_continue(state: AgentState) -> str:
    """Decide whether to call tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ── Graph ─────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("scientist", scientist_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_plots", extract_plots)

graph.set_entry_point("scientist")
graph.add_conditional_edges("scientist", should_continue, {
    "tools": "tools",
    "end": END,
})
graph.add_edge("tools", "scientist")

app = graph.compile()

def run_agent(question: str) -> dict:
    """Run the agent with a clinical question."""
    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "plots": []
    })
    
    # Extract final response
    final_message = result["messages"][-1].content
    if isinstance(final_message, list):
        text = " ".join([b.get("text", "") for b in final_message 
                        if isinstance(b, dict)])
    else:
        text = str(final_message)
    
    # Extract plots from all tool results
    plots = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                try:
                    data = ast.literal_eval(msg.content)
                except:
                    data = {}
            if isinstance(data, dict) and "plot_base64" in data:
                plots.append(data["plot_base64"])

    return {
        "response": text,
        "plots": plots,
        "messages": result["messages"]
    }