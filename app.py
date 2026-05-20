"""
app.py — Streamlit UI for the Oncology RWE Co-Scientist Agent
"""

import streamlit as st
import base64
from PIL import Image
import io
from src.agent import run_agent
import time


st.set_page_config(
    page_title="Oncology RWE Co-Scientist",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #2E4057;
    border-color: #2E4057;
    color: white;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #048A81;
    border-color: #048A81;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Oncology RWE Co-Scientist")
st.markdown("""
An AI-powered research agent that analyzes real-world breast cancer survival data
using Kaplan-Meier curves, Cox PH models, and ML prediction.

**Dataset:** METABRIC (n=496 patients, median follow-up 7.8 years)
""")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Example Questions")
    examples = [
    "Describe the dataset",
    "What clinical factors are most prognostic for survival in this cohort?",
    "Run a Kaplan-Meier analysis stratified by subtype.",
    "Is hormone therapy associated with better survival? Control for confounders.",
    "Predict 5-year mortality risk using machine learning",
    "What color is the sky?",  # tests domain boundary
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

    st.divider()
    st.markdown("**Data source:** [METABRIC via cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)")
    st.markdown("**Agent:** Claude Sonnet + LangGraph")

# ── Main input ─────────────────────────────────────────────────────────────────
question = st.text_area(
    "Ask a clinical research question:",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. What factors predict breast cancer survival in this cohort?"
)

if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("🤖 Co-scientist is analyzing..."):
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = run_agent(question)
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                        placeholder = st.empty()
                        for remaining in range(60, 0, -1):
                            placeholder.warning(f"⏳ Rate limit reached — retrying in {remaining}s...")
                            time.sleep(1)
                        placeholder.empty()
                    elif "rate_limit" in str(e).lower():
                        st.error("⏳ Rate limit reached — please try again in a moment.")
                        st.stop()
                    else:
                        st.error(f"An error occurred: {str(e)}")
                        st.stop()

        st.divider()

        # ── Response ───────────────────────────────────────────────────────────
        st.subheader("📋 Scientific Report")
        st.markdown(result["response"])

        # ── Plots ─────────────────────────────────────────────────────────────
        if result["plots"]:
            st.subheader("📊 Figures")
            cols = st.columns(min(len(result["plots"]), 2))
            for i, plot_b64 in enumerate(result["plots"]):
                img_data = base64.b64decode(plot_b64)
                img = Image.open(io.BytesIO(img_data))
                cols[i % 2].image(img, width=600)

        # ── Download report ────────────────────────────────────────────────────
        st.divider()

        plots_html = ""
        for i, plot_b64 in enumerate(result["plots"]):
            plots_html += f"""
            <div style="margin: 20px 0;">
                <h3>Figure {i+1}</h3>
                <img src="data:image/png;base64,{plot_b64}"
                     style="max-width:100%; border:1px solid #ddd; border-radius:4px;">
            </div>
            """

        report_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Oncology RWE Analysis Report</title>
    <style>
        body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto;
                padding: 0 20px; color: #2E4057; }}
        h1 {{ color: #2E4057; border-bottom: 2px solid #048A81; padding-bottom: 10px; }}
        h2 {{ color: #048A81; }}
        .meta {{ color: #6C757D; font-size: 0.9em; margin-top: 40px;
                 border-top: 1px solid #ddd; padding-top: 10px; }}
    </style>
</head>
<body>
    <h1>🔬 Oncology RWE Analysis Report</h1>
    <h2>Research Question</h2>
    <p>{question}</p>
    <h2>Scientific Findings</h2>
    {result["response"].replace(chr(10), "<br>")}
    <h2>Figures</h2>
    {plots_html if plots_html else "<p>No figures generated.</p>"}
    <div class="meta">
        <p>Generated by Oncology RWE Co-Scientist — METABRIC dataset (n=496)<br>
        Agent: Claude Sonnet + LangGraph</p>
    </div>
</body>
</html>"""

        has_figures = len(result["plots"]) > 0

        st.download_button(
            label="⬇️ Download Report (HTML with figures)" if has_figures else "⬇️ Download Report (HTML)",
            data=report_html,
            file_name="rwe_analysis_report.html",
            mime="text/html",
            use_container_width=True
        )

        # ── Agent trace ────────────────────────────────────────────────────────
        with st.expander("🔎 Agent reasoning trace"):
            for msg in result["messages"]:
                role = type(msg).__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        st.info(f"🔧 **Tool called:** `{tc['name']}`  \n**Input:** `{tc['args']}`")
                elif hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        text = " ".join([b.get("text", "") for b in content
                                        if isinstance(b, dict) and "text" in b])
                    else:
                        text = str(content)
                    if text.strip() and role != "ToolMessage":
                        st.markdown(f"**{role}:** {text[:500]}...")
