import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Academic Emotion Analytics Workspace")

# 1. Color Palette Mapping to Match UI Exactly
EMOTION_COLORS = {
    "Bored": "#1F77B4",       # Blue
    "Confident": "#FF7F0E",   # Orange
    "Confused": "#2CA02C",    # Green / Teal
    "Curious": "#9467BD",     # Purple
    "Frustrated": "#8C564B"   # Brownish Red
}

# 2. Session State Initialization
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = [
        {"timestamp": "21:00:00", "field": "Computer Science", "emotion": "Curious", "confidence": 0.6},
        {"timestamp": "22:37:13", "field": "Computer Science", "emotion": "Curious", "confidence": 0.6},
        {"timestamp": "22:37:58", "field": "Software Engineering", "emotion": "Bored", "confidence": 0.2}
    ]

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

# 3. Application Layout Headers
st.title("🎓 Academic Emotion Analytics Workspace")

# Main Application Tabs
app_tab1, app_tab2, app_tab3 = st.tabs(["📊 Input Interface", "📈 Analysis Results", "📜 Historical Analytics"])

# --- TAB 1: INPUT INTERFACE & SETTINGS ---
with app_tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Capture Live Student Interaction Insights")
        
        department = st.selectbox(
            "Select Active Department Field Mode:",
            ["Computer Science", "Software Engineering", "Data Science", "Information Technology"]
        )
        
        phrase_input = st.text_area(
            "Enter student phrase query or log stack details:",
            value="today i got more marks in todays exams"
        )
        
        if st.button("Run Model Prediction", type="primary"):
            # Mock evaluation logic to match the requested snapshot context
            if "more marks" in phrase_input.lower() or "today i got more marks in todays exams" in phrase_input.lower():
                pred_emotion = "Confident"
                confidence = 0.95
                probabilities = {"Bored": 0.01, "Confident": 0.95, "Confused": 0.02, "Curious": 0.01, "Frustrated": 0.01}
            else:
                pred_emotion = "Curious"
                confidence = 0.70
                probabilities = {"Bored": 0.05, "Confident": 0.10, "Confused": 0.10, "Curious": 0.70, "Frustrated": 0.05}
            
            # Save current output metrics
            st.session_state.current_prediction = {
                "phrase": phrase_input,
                "winner": pred_emotion,
                "confidence": confidence,
                "probabilities": probabilities
            }
            
            # Append to history logs
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.emotion_history.append({
                "timestamp": now,
                "field": department,
                "emotion": pred_emotion,
                "confidence": confidence
            })
            
            st.success("✅ Processed! Switch to the 'Analysis Results' tab now to view the score charts.")
            
    with col2:
        st.subheader("⚙️ Settings")
        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_data = st.checkbox("Save to CSV for learning", value=True)
        show_details = st.checkbox("Show analysis details", value=False)
        
        st.markdown("---")
        st.write("**Predict from Saved Data**")
        use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)
        
        # Creating mock examples dataframe for sidebar interaction context
        examples_df = pd.DataFrame(st.session_state.emotion_history)
        if use_csv_prediction and len(examples_df) > 0:
            st.info(f"Using {len(examples_df)} saved examples for prediction")

# --- TAB 2: ANALYSIS RESULTS ---
with app_tab2:
    if st.session_state.current_prediction is not None:
        cp = st.session_state.current_prediction
        
        st.subheader("Live Inference Metrics Breakdown")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.write("Primary Category Winner State Mode")
            st.markdown(f"## **{cp['winner']}**")
        with metric_col2:
            st.write("Ensemble Voting Confidence Score")
            st.markdown(f"## **{cp['confidence']*100:.2f}%**")
            
        st.info(f"Processed Phrase Context: \"{cp['phrase']}\"")
        
        # Build Dataframe for Probabilities Bar Chart
        chart_data = pd.DataFrame({
            "Emotion": list(cp["probabilities"].keys()),
            "Probability": list(cp["probabilities"].values())
        })
        
        # Render Plotly Bar Chart with Custom Assigned Hex Colors
        fig_bar = px.bar(
            chart_data, 
            x="Emotion", 
            y="Probability", 
            color="Emotion",
            color_discrete_map=EMOTION_COLORS,
            text=chart_data["Probability"].apply(lambda x: f"{x*100:.2f}%")
        )
        fig_bar.update_layout(yaxis_range=[0, 1], showlegend=True, height=450)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No dynamic evaluation runs recorded yet. Go to 'Input Interface' and click 'Run Model Prediction'.")

# --- TAB 3: HISTORICAL ANALYTICS ---
with app_tab3:
    if len(st.session_state.emotion_history) > 0:
        st.subheader("Historical Analytics Workspace Overview")
        
        df_history = pd.DataFrame(st.session_state.emotion_history)
        
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            # Donut Chart for distribution
            fig_donut = px.pie(
                df_history, 
                names="emotion", 
                hole=0.4,
                color="emotion",
                color_discrete_map=EMOTION_COLORS,
                title="Emotion Metrics Distribution Breakdown"
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with hist_col2:
            # Line Chart for Confidence Timelines
            fig_line = px.line(
                df_history, 
                x="timestamp", 
                y="confidence", 
                color="emotion",
                markers=True,
                color_discrete_map=EMOTION_COLORS,
                title="Confidence Trend Track Over Time"
            )
            fig_line.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_line, use_container_width=True)
            
        # Dataframe Log Table View
        st.markdown("### Historical Session Evaluation Logs Dataframe")
        st.dataframe(df_history, use_container_width=True)
        
    else:
        st.info("No active history available.")