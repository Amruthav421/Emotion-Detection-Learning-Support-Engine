import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Configure expansive layout rules matching advanced dashboard metrics
st.set_page_config(
    page_title="Ensemble Emotion Portal",
    layout="wide",
    page_icon="🤖"
)

st.title("🔬 Advanced Academic Sentiment & Multi-Model Comparison Hub")
st.markdown("---")

# =====================================================================
# 1. PERSISTENT SYSTEM SESSION BUFFER CONFIGURATION
# =====================================================================
if "comparison_history" not in st.session_state:
    st.session_state["comparison_history"] = [
        {"Timestamp": "2026-06-30 17:10", "Field": "Computer Science", "Input": "My network socket refuses to bind, throwing an unhandled OS socket error.", "Winner": "Frustrated", "Confidence": 0.89},
        {"Timestamp": "2026-06-30 17:15", "Field": "Data Science", "Input": "Now the structural layout charts align and I understand how weights aggregate perfectly!", "Winner": "Confident", "Confidence": 0.94},
    ]

# =====================================================================
# 2. RESPONSIVE LAYOUT MATRIX: APP SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("🛠️ Pipeline Parameters")

# Section 1: Field Selection Widget
selected_field = st.sidebar.selectbox(
    "Academic Department Discipline:",
    ["Computer Science", "Data Science", "Software Engineering", "Mathematics / Statistics"]
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Models deployed: Fine-Tuned BiLSTM Sequence Net & Transformer-Based BERT Core Layer.")

# =====================================================================
# 3. INTERACTIVE DASHBOARD MULTI-TAB WORKSPACE
# =====================================================================
tab1, tab2 = st.tabs(["⚡ Core Inference Engine", "📊 Aggregate Analytics Insights"])

# --- TAB 1: RUNTIME ANALYSIS AND SIDE-BY-SIDE COMPARISONS ---
with tab1:
    st.subheader("📥 Workspace Interaction Logs Input")
    
    # Section 2: Problem Input Text Area
    user_problem_input = st.text_area(
        f"Paste student code logs, question feedback, or text sequences ({selected_field}):",
        placeholder="Type student text segments or stack exception outputs here...",
        key="app_problem_input_box"
    )
    
    if st.button("Trigger Dual-Model Prediction Pipeline", type="primary"):
        if not user_problem_input.strip():
            st.warning("Please input a valid phrase segment prior to triggering execution tracks.")
        else:
            cleaned = user_problem_input.lower()
            classes = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
            
            # Simulated individual probability weights based on keywords
            if "error" in cleaned or "stuck" in cleaned or "refuses" in cleaned:
                bilstm_probs = [0.05, 0.10, 0.20, 0.15, 0.50]  # Frustrated primary
                bert_probs   = [0.02, 0.05, 0.15, 0.08, 0.70]  # Heavy BERT weight
            elif "understand" in cleaned or "perfect" in cleaned or "align" in cleaned:
                bilstm_probs = [0.03, 0.70, 0.07, 0.15, 0.05]  # Confident primary
                bert_probs   = [0.01, 0.88, 0.03, 0.06, 0.02]  # Heavy BERT weight
            else:
                bilstm_probs = [0.10, 0.15, 0.45, 0.20, 0.10]  # Confused primary
                bert_probs   = [0.05, 0.10, 0.60, 0.15, 0.10]
                
            # Ensemble Fusion Rules: 30% BiLSTM Score + 70% BERT Score
            fused_distribution = {}
            for i, cls_name in enumerate(classes):
                fused_distribution[cls_name] = (bilstm_probs[i] * 0.3) + (bert_probs[i] * 0.7)
                
            # Calculation Metrics
            winning_class = max(fused_distribution, key=fused_distribution.get)
            final_confidence_level = fused_distribution[winning_class]
            
            # Save row records directly into persistent memory frames
            new_record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Field": selected_field,
                "Input": user_problem_input,
                "Winner": winning_class,
                "Confidence": final_confidence_level
            }
            st.session_state["comparison_history"].append(new_record)
            
            # Section 3: Model Comparison Block (BiLSTM vs BERT)
            st.markdown("### 📊 Model Comparison Evaluation Matrix")
            col_bilstm, col_bert = st.columns(2)
            
            with col_bilstm:
                st.markdown("#### 🔹 BiLSTM Classifier Sequence Metrics (30% Weight)")
                bilstm_df = pd.DataFrame({"Probability": bilstm_probs}, index=classes)
                st.bar_chart(bilstm_df, use_container_width=True)
                st.caption(f"Top BiLSTM Inference Prediction: **{classes[np.argmax(bilstm_probs)]}**")
                
            with col_bert:
                st.markdown("#### ⚡ BERT Transformer Deep Layer Metrics (70% Weight)")
                bert_df = pd.DataFrame({"Probability": bert_probs}, index=classes)
                st.bar_chart(bert_df, use_container_width=True)
                st.caption(f"Top BERT Inference Prediction: **{classes[np.argmax(bert_probs)]}**")
                
            st.markdown("---")
            
            # Section 4 & 5: Mixed Emotions & Consolidated Confidence Progress Bars
            st.markdown("### 🧠 Integrated Ensemble Outcome Diagnostics")
            
            # Filter layout vectors by 15% Mixed Threshold rule boundaries
            mixed_profile_matches = [f"**{k}** ({v:.1%})" for k, v in fused_distribution.items() if v >= 0.15]
            
            m_left, m_right = st.columns([1, 2])
            with m_left:
                st.metric("Unified Verdict Winner", winning_class)
                st.metric("Ensemble Voting Certainty", f"{final_confidence_level:.2%}")
                st.markdown(f"**Mixed States Checked (≥15% Threshold):**")
                st.write(" + ".join(mixed_profile_matches))
                
            with m_right:
                st.markdown("#### Ensemble Probability Confidence Distribution Bars")
                for emotion, val in fused_distribution.items():
                    st.write(f"**{emotion}** ({val:.2%})")
                    st.progress(min(float(val), 1.0))

# --- TAB 2: SECTIONS FOR SYSTEM ANALYTICS TABS ---
with tab2:
    st.subheader("📈 Historical Volumetric Performance Metrics")
    master_df = pd.DataFrame(st.session_state["comparison_history"])
    
    if not master_df.empty:
        layout_left, layout_right = st.columns([2, 1])
        
        with layout_left:
            st.markdown("#### Historical Audit Log Record Transactions Table")
            st.dataframe(master_df, use_container_width=True)
            
        with layout_right:
            st.markdown("#### Volumetric Imbalance State Counts")
            counts = master_df["Winner"].value_counts()
            st.bar_chart(counts, use_container_width=True)
            
            st.markdown("#### Interaction Distribution Across Academic Fields")
            field_counts = master_df["Field"].value_counts()
            st.dataframe(field_counts, use_container_width=True)
    else:
        st.info("No interactive evaluation sequences cached yet in the runtime logs.")