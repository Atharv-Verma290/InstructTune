import streamlit as st
import pandas as pd
import plotly.express as px
from main import graph  # this imports the graph from your main.py

st.set_page_config(page_title="PromptPilot", layout="wide")

st.title("🧪 PromptPilot - Prompt Evaluation & Optimization")

with st.sidebar:
    st.header("Prompt Setup")
    original_prompt = st.text_area("Prompt", "You are a teacher. Teach the given topic.")
    input_example = st.text_area("Input", "What is gravity?")
    context = st.text_area("Context", "Middle school teaching agent tasked with explaining complex concepts to simple terms. For teaching middle school students as a teaching assistant.")
    run_button = st.button("🚀 Run Prompt Evaluation")

if run_button:
    with st.spinner("Running PromptPilot workflow..."):
        state = graph.invoke({
            "original_prompt": original_prompt,
            "input_example": input_example,
            "context": context,
            "candidate_prompts": [original_prompt]
        })

    st.success("Done! Here's your report 👇")

    # -- PROMPT HISTORY --
    st.header("📜 Prompt Versions")
    for i, prompt in enumerate(state.get("candidate_prompts", [])):
        label = "Original" if i == 0 else f"Optimized {i}"
        st.code(prompt, language="markdown")
        st.markdown(f"**Version {label}**")

    # -- MODEL OUTPUTS --
    st.header("📤 Model Outputs")
    outputs = state.get("outputs", {})
    for model, versions in outputs.items():
        for version, result in versions.items():
            with st.expander(f"🧠 {model.upper()} - {version}"):
                st.text_area("Output", result.get("output", ""), height=150)
                st.markdown(f"**Latency**: {result.get('latency', 'N/A')} s")
                st.markdown(f"**Tokens**: {result.get('tokens', 'N/A')}")
                st.markdown(f"**Cost**: ${result.get('cost', 0):.4f}")

    # -- EVALUATION SCORES --
    st.header("📊 Evaluation Scores")
    rows = []
    evals = state.get("evaluations", {})
    for model, versions in evals.items():
        for version, scores in versions.items():
            row = {"Model": model.upper(), "Version": version}
            row.update(scores)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)

        # Optional: Metric visualizations
        st.subheader("📈 Metric Comparison")
        metrics = [col for col in df.columns if col not in ["Model", "Version", "feedback"]]
        for metric in metrics:
            fig = px.bar(
                df, x="Model", y=metric, color="Version",
                barmode="group", title=f"{metric.title()} Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Optional: Radar chart for one metric group
        if len(df) > 1:
            radar_data = df.melt(id_vars=["Model", "Version"], value_vars=metrics)
            fig = px.line_polar(
                radar_data,
                r="value", theta="variable",
                color="Model",
                line_group="Version",
                title="Radar Chart - Model Performance"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No evaluation results found.")

    # -- FINAL OPTIMIZED PROMPT --
    if state.get("optimized_prompt"):
        st.header("🎯 Optimized Prompt")
        st.code(state["optimized_prompt"], language="markdown")

else:
    st.info("Enter a prompt and context, then click 'Run Prompt Evaluation'.")
