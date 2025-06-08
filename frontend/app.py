import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_URL = "http://backend:8000/run-graph"

st.set_page_config(page_title="InstructTune", layout="wide")
st.title("InstructTune – Agent Instruction Optimization & Evaluation")

# --- Sidebar Form for Input ---
with st.sidebar.form("model_selection_form"):
    st.header("Prompt Configuration")

    instruction = st.text_area("Agent Instruction Prompt", "Teach given user topic.")
    input_example = st.text_area("Input Example", "What is gravity?")
    # output_example = st.text_area("Expected Output", "Great camera, average battery.") ignore Expected Output for now.
    context = st.text_area("Context / Problem Description", "A middle school teaching assistant tasked with teach complex topics to middle school kids.")

    st.subheader("Select up to 3 models")

    models_list = []
    for i in range(1, 4):
        st.markdown(f"**Model {i}**")
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox(f"Provider {i}", ["", "openai", "google-genai"], key=f"provider_{i}")
        with col2:
            model = st.text_input(f"Model Name {i}", key=f"model_{i}")

        if provider and model:
            models_list.append({model: provider})  # <-- changed from dict to list of dicts

    submitted = st.form_submit_button("🚀 Run Instruction Evaluation")

# --- Run Graph ---
if submitted and models_list:
    with st.spinner("Running instruction evaluation workflow..."):
        payload = {
            "original_prompt": instruction,
            "input_example": input_example,
            "context": context,
            "candidate_prompts": [instruction],
            "models_list": models_list
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            state = response.json()
            st.success("Instruction prompt evaluation complete!")
        except Exception as e:
            st.error(f"API call failed: {e}")
            state = {}

    # --- Show Prompt Versions ---
    st.header("📜 Instruction Prompt Versions")
    for i, prompt in enumerate(state.get("candidate_prompts", [])):
        label = "Original" if i == 0 else f"Optimized {i}"
        st.code(prompt, language="markdown")
        st.markdown(f"**Version: {label}**")

    # --- Model Outputs ---
    st.header("📤 Model Outputs")
    for model, versions in state.get("outputs", {}).items():
        for version, data in versions.items():
            with st.expander(f"🧠 {model.upper()} – {version}"):
                st.text_area("Output", data.get("output", ""), height=150)
                st.markdown(f"**Latency**: {data.get('latency', 'N/A')} s")
                st.markdown(f"**Tokens**: {data.get('tokens', 'N/A')}")
                # st.markdown(f"**Cost**: ${data.get('cost', 0):.4f}") cost is ignored for now

    # --- Evaluation Scores ---
    st.header("📊 Evaluation Scores")
    rows = []
    for model, versions in state.get("evaluations", {}).items():
        for version, scores in versions.items():
            row = {"Model": model, "Version": version}
            row.update(scores)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)

        st.subheader("📈 Metric Comparison")
        metric_cols = [col for col in df.columns if col not in ["Model", "Version", "feedback"]]
        for metric in metric_cols:
            fig = px.bar(df, x="Model", y=metric, color="Version", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    # --- Optimized Prompt (if available) ---
    if state.get("optimized_prompt"):
        st.header("🎯 Optimized Instruction")
        st.code(state["optimized_prompt"], language="markdown")

else:
    if submitted and not models_list:
        st.warning("Please select at least one model and provider.")
    else:
        st.info("Fill the prompt and model details, then click **Run Instruction Prompt Evaluation**.")
