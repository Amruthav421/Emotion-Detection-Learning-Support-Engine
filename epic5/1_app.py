import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =====================================================================
# SYSTEM PATH PATH CONFIGURATION (Fixes ModuleNotFoundError)
# =====================================================================
# Dynamically locate root workspace level and append to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Configure wide responsive page settings
st.set_page_config(
    page_title="Academic Emotion Dashboard",
    layout="wide",
    page_icon="🎓"
)

st.title("🎓 Academic Emotion Analytics Workspace")
st.markdown("---")

# =====================================================================
# 1. INITIALIZE PERSISTENT SESSION STATE TRACKING
# =====================================================================
if "history" not in st.session_state:
    st.session_state.history = [
        {"Timestamp": "2026-06-30 14:10", "Text Input": "I cannot get this specific model to finish compiling.", "Predicted Emotion": "Frustrated", "Confidence": 0.8540},
        {"Timestamp": "2026-06-30 14:15", "Text Input": "Now the instructions make perfect sense!", "Predicted Emotion": "Confident", "Confidence": 0.9210},
        {"Timestamp": "2026-06-30 14:22", "Text Input": "What exactly does this parameter argument optimize?", "Predicted Emotion": "Confused", "Confidence": 0.7830}
    ]

# =====================================================================
# 2. RESPONSIVE THREE-TAB CONTENT LAYOUT 
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📥 Input Interface", "🎯 Analysis Results", "📊 Historical Analytics"])

# --- TAB 1: STUDENT DATA INPUTS ---
with tab1:
    st.subheader("Capture Live Student Interaction Insights")
    user_text = st.text_area(
        "Enter student phrase query or log stack details:",
        placeholder="Type here to route statements into classification engines...",
        key="student_input_text_area"
    )
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        submit_btn = st.button("Run Model Prediction", type="primary")
        
    if submit_btn and user_text.strip():
        # Heuristic rules to assign mock values for testing
        cleaned = user_text.lower()
        if "stuck" in cleaned or "error" in cleaned or "fail" in cleaned:
            predicted = "Frustrated"
            confidence = float(np.random.uniform(0.80, 0.95))
        elif "clear" in cleaned or "understand" in cleaned or "perfect" in cleaned:
            predicted = "Confident"
            confidence = float(np.random.uniform(0.85, 0.98))
        else:
            predicted = "Confused"
            confidence = float(np.random.uniform(0.60, 0.79))
            
        # Append directly to state memory frame array
        new_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Text Input": user_text,
            "Predicted Emotion": predicted,
            "Confidence": confidence
        }
        st.session_state.history.append(new_entry)
        
        # Save the current execution index for the Result viewing tab
        st.session_state["latest_index"] = len(st.session_state.history) - 1
        st.success("🎉 Phrase analyzed! Head over to the 'Analysis Results' tab to look at diagnostic charts.")

# --- TAB 2: CURRENT RESULTS VISUALIZATION ---
with tab2:
    st.subheader("Live Inference Metrics Breakdown")
    if len(st.session_state.history) > 0:
        # Pull the absolute latest interaction event block
        target_idx = st.session_state.get("latest_index", len(st.session_state.history) - 1)
        latest_record = st.session_state.history[target_idx]
        
        # Grid metrics columns split
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Primary Category Winner", latest_record["Predicted Emotion"])
        with m_col2:
            st.metric("Model Confidence Boundary", f"{latest_record['Confidence']:.2%}")
        with m_col3:
            st.metric("System Timestamp", latest_record["Timestamp"])
            
        st.info(f"**Target Raw Input String Context:** \"{latest_record['Text Input']}\"")
        
        # Render confidence graph vectors
        st.markdown("#### Probability Vector Balance Chart")
        emotions_list = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
        
        # Distribute fake weight profiles matching the primary class outcome
        if latest_record["Predicted Emotion"] == "Frustrated":
            scores_vec = [0.05, 0.05, 0.10, 0.10, 0.70]
        elif latest_record["Predicted Emotion"] == "Confident":
            scores_vec = [0.02, 0.75, 0.03, 0.15, 0.05]
        else:
            scores_vec = [0.10, 0.10, 0.60, 0.10, 0.10]
            
        chart_df = pd.DataFrame({
            "Emotion": emotions_list,
            "Score": scores_vec
        }).set_index("Emotion")
        
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.info("Awaiting live processing requests inside the Input tab.")

# --- TAB 3: ANALYTICS HISTORY DATA VIEWS ---
with tab3:
    st.subheader("Historical System Transaction Log Analytics")
    history_df = pd.DataFrame(st.session_state.history)
    
    if not history_df.empty:
        # Split layout view windows side by side
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.markdown("#### System Session Operations Table")
            st.dataframe(history_df, use_container_width=True)
            
        with right_col:
            st.markdown("#### Cumulative Category Distribution")
            class_counts = history_df["Predicted Emotion"].value_counts()
            st.bar_chart(class_counts, use_container_width=True)
    else:
        st.caption("No tracking data rows detected.")