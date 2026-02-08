"""
Workshop Attendance Prediction Dashboard
=========================================
Interactive dashboard for predicting workshop attendance 
and analyzing engagement patterns.

Run: streamlit run app.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.predict import predict_single_event, load_latest_model
from src.feature_engineering import run_feature_pipeline

MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---- Page Config ----
st.set_page_config(
    page_title="Workshop Attendance Predictor",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    return pd.read_csv(csv_path)


@st.cache_data
def load_feature_importance():
    path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_model_info():
    """Load metadata for the best model (highest F1)"""
    best_meta = None
    best_f1 = -1
    for name in ["xgboost", "random_forest"]:
        meta_path = os.path.join(MODELS_DIR, f"{name}_latest_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f1 = meta.get('metrics', {}).get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_meta = meta
    return best_meta


# ---- Sidebar Navigation ----
st.sidebar.title("📊 Workshop Predictor")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔮 Predict Attendance",
    "📈 Attendance Trends",
    "🔍 Topic Analysis",
    "⚙️ Model Performance"
])


# ---- Load Data ----
df = load_data()
df['event_date'] = pd.to_datetime(df['event_date'])


# ========================================================
# PAGE: Overview
# ========================================================
if page == "🏠 Overview":
    st.title("Workshop Attendance Prediction System")
    st.markdown("*ML-powered predictions to help organizers plan better events*")
    
    # key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", df['event_id'].nunique())
    with col2:
        st.metric("Total Students", df['student_id'].nunique())
    with col3:
        st.metric("Total Registrations", len(df))
    with col4:
        st.metric("Avg Attendance Rate", f"{df['attended'].mean():.1%}")
    
    st.markdown("---")
    
    # quick summary charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Attendance by Topic")
        topic_rates = df.groupby('topic')['attended'].agg(['mean', 'count']).sort_values('mean', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(topic_rates.index, topic_rates['mean'], color='#4C72B0')
        ax.set_xlabel("Attendance Rate")
        ax.set_title("Which topics attract more students?")
        for bar, count in zip(bars, topic_rates['count']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f"n={count}", va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_right:
        st.subheader("Attendance by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_rates = df.groupby('day_of_week')['attended'].mean().reindex(day_order)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#E74C3C' if d in ['Saturday', 'Sunday'] else '#4C72B0' for d in day_order]
        ax.bar(day_rates.index, day_rates.values, color=colors)
        ax.set_ylabel("Attendance Rate")
        ax.set_title("Weekday vs Weekend attendance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # model info
    model_info = load_model_info()
    if model_info:
        st.markdown("---")
        st.subheader("Active Model")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Model", model_info['model_name'].upper())
        with mc2:
            st.metric("F1 Score", f"{model_info['metrics']['f1_score']:.3f}")
        with mc3:
            st.metric("AUC-ROC", f"{model_info['metrics']['auc_roc']:.3f}")


# ========================================================
# PAGE: Predict Attendance
# ========================================================
elif page == "🔮 Predict Attendance":
    st.title("Predict Workshop Attendance")
    st.markdown("Enter details of your planned workshop to get a prediction.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.selectbox("Workshop Topic", sorted(df['topic'].unique()))
            speaker_type = st.selectbox("Speaker Type", sorted(df['speaker_type'].unique()))
            day_of_week = st.selectbox("Day of Week", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            time_slot = st.selectbox("Time Slot", sorted(df['time_slot'].unique()))
        
        with col2:
            mode = st.selectbox("Mode", ['Offline', 'Online'])
            duration = st.slider("Duration (minutes)", 30, 240, 90, step=30)
            num_registrations = st.slider("Expected Registrations", 10, 150, 50)
            exam_proximity = st.select_slider("Exam Proximity", 
                options=[1, 2, 3], 
                format_func=lambda x: {1: "Near Exams", 2: "Moderate", 3: "Far from Exams"}[x])
            promotion_level = st.selectbox("Promotion Level", ['Low', 'Medium', 'High'])
        
        event_date = st.date_input("Event Date")
        submitted = st.form_submit_button("🔮 Predict Attendance", use_container_width=True)
    
    if submitted:
        event_details = {
            'topic': topic,
            'speaker_type': speaker_type,
            'day_of_week': day_of_week,
            'time_slot': time_slot,
            'duration_minutes': duration,
            'mode': mode,
            'exam_proximity': exam_proximity,
            'promotion_level': promotion_level,
            'num_registrations': num_registrations,
            'event_date': str(event_date)
        }
        
        with st.spinner("Running prediction..."):
            try:
                result = predict_single_event(event_details, df)
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Registered", result['total_registered'])
                with r2:
                    st.metric("Predicted Attendees", result['predicted_attendees'])
                with r3:
                    st.metric("Predicted Rate", f"{result['attendance_rate']:.1%}")
                with r4:
                    conf_color = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
                    st.metric("Confidence", f"{conf_color.get(result['confidence'], '')} {result['confidence']}")
                
                # recommendation
                st.markdown("---")
                st.subheader("Recommendations")
                attendees = result['predicted_attendees']
                if attendees < 20:
                    st.warning("⚠️ Low predicted turnout. Consider changing the topic, timing, or boosting promotion.")
                elif attendees < 40:
                    st.info("ℹ️ Moderate turnout expected. A medium-sized room should work.")
                else:
                    st.success("✅ Good turnout expected! Prepare a larger venue.")
                
                # historical comparison
                topic_history = df[df['topic'] == topic]
                if len(topic_history) > 0:
                    hist_rate = topic_history['attended'].mean()
                    st.markdown(f"**Historical reference:** *{topic}* workshops usually have "
                               f"**{hist_rate:.1%}** attendance rate across {topic_history['event_id'].nunique()} past events.")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Make sure you've trained the model first (run: python main.py)")


# ========================================================
# PAGE: Attendance Trends
# ========================================================
elif page == "📈 Attendance Trends":
    st.title("Attendance Trends Over Time")
    
    # monthly attendance trend
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['event_date'].dt.to_period('M').astype(str)
    monthly_stats = df_monthly.groupby('month').agg(
        total_registrations=('attended', 'count'),
        total_attended=('attended', 'sum'),
        attendance_rate=('attended', 'mean')
    ).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(monthly_stats['month'], monthly_stats['total_registrations'], 
             marker='o', label='Registrations', color='#3498DB')
    ax1.plot(monthly_stats['month'], monthly_stats['total_attended'], 
             marker='s', label='Actually Attended', color='#2ECC71')
    ax1.set_title("Monthly Registrations vs Actual Attendance")
    ax1.legend()
    ax1.set_ylabel("Count")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    ax2.bar(monthly_stats['month'], monthly_stats['attendance_rate'], color='#E67E22')
    ax2.set_title("Monthly Attendance Rate")
    ax2.set_ylabel("Rate")
    ax2.axhline(y=df['attended'].mean(), color='red', linestyle='--', label='Overall Average')
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # attendance by various factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impact of Exam Proximity")
        exam_data = df.groupby('exam_proximity')['attended'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Near (1)', 'Moderate (2)', 'Far (3)'], exam_data.values, 
                      color=['#E74C3C', '#F39C12', '#2ECC71'])
        ax.set_ylabel("Attendance Rate")
        ax.set_title("Students skip more when exams are near")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.1%}', ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Impact of Speaker Type")
        speaker_data = df.groupby('speaker_type')['attended'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(speaker_data.index, speaker_data.values, color='#9B59B6')
        ax.set_xlabel("Attendance Rate")
        ax.set_title("Industry speakers draw the most students")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # time slot analysis
    st.subheader("Best Time Slots")
    slot_day = df.groupby(['day_of_week', 'time_slot'])['attended'].mean().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    slot_day = slot_day.reindex(day_order)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(slot_day, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax)
    ax.set_title("Attendance Rate: Day vs Time Slot")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ========================================================
# PAGE: Topic Analysis
# ========================================================
elif page == "🔍 Topic Analysis":
    st.title("Topic & Event Analysis")
    
    # topic selector
    selected_topic = st.selectbox("Select Topic to Analyze", 
                                  ['All Topics'] + sorted(df['topic'].unique().tolist()))
    
    if selected_topic != 'All Topics':
        topic_df = df[df['topic'] == selected_topic]
    else:
        topic_df = df
    
    # stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Events", topic_df['event_id'].nunique())
    with col2:
        st.metric("Registrations", len(topic_df))
    with col3:
        st.metric("Attended", topic_df['attended'].sum())
    with col4:
        st.metric("Rate", f"{topic_df['attended'].mean():.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attendance by Department")
        dept_data = topic_df.groupby('department')['attended'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(dept_data.index, dept_data.values, color='#1ABC9C')
        ax.set_xlabel("Attendance Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Attendance by Semester")
        sem_data = topic_df.groupby('semester')['attended'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sem_data.index, sem_data.values, color='#3498DB')
        ax.set_xlabel("Semester")
        ax.set_ylabel("Attendance Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # mode comparison
    st.subheader("Online vs Offline Comparison")
    mode_data = topic_df.groupby('mode').agg(
        count=('attended', 'count'),
        attended=('attended', 'sum'),
        rate=('attended', 'mean')
    ).reset_index()
    
    mc1, mc2 = st.columns(2)
    for i, (_, row) in enumerate(mode_data.iterrows()):
        with [mc1, mc2][i]:
            st.metric(f"{row['mode']}", f"{row['rate']:.1%}", 
                     f"{int(row['attended'])}/{int(row['count'])} attended")
    
    # club activity impact
    st.markdown("---")
    st.subheader("Does Club Activity Level Matter?")
    club_data = topic_df.groupby('club_activity_level')['attended'].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {'Low': '#E74C3C', 'Medium': '#F39C12', 'High': '#2ECC71'}
    for level in ['Low', 'Medium', 'High']:
        if level in club_data:
            ax.bar(level, club_data[level], color=colors[level])
    ax.set_ylabel("Attendance Rate")
    ax.set_title("Higher club activity = slightly higher attendance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ========================================================
# PAGE: Model Performance
# ========================================================
elif page == "⚙️ Model Performance":
    st.title("Model Performance & Details")
    
    model_info = load_model_info()
    
    if model_info:
        st.subheader("Current Active Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_info['model_name'].upper())
        with col2:
            st.metric("Accuracy", f"{model_info['metrics']['accuracy']:.3f}")
        with col3:
            st.metric("F1 Score", f"{model_info['metrics']['f1_score']:.3f}")
        with col4:
            st.metric("AUC-ROC", f"{model_info['metrics']['auc_roc']:.3f}")
        
        st.markdown("---")
        
        # feature importance
        feat_imp = load_feature_importance()
        if feat_imp is not None:
            st.subheader("Top 15 Most Important Features")
            top_15 = feat_imp.head(15).sort_values('importance', ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_15['feature'], top_15['importance'], color='#2980B9')
            ax.set_xlabel("Importance Score")
            ax.set_title("What factors matter most for attendance?")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # explanation
        st.markdown("---")
        st.subheader("How the Model Works")
        active_name = model_info['model_name'].upper()
        if model_info['model_name'] == 'xgboost':
            st.markdown(f"""
            **Algorithm:** XGBoost (Extreme Gradient Boosting)
            
            XGBoost builds multiple decision trees sequentially, where each new tree 
            tries to fix the mistakes of the previous ones. This is called "boosting."
            
            **Why XGBoost for this problem:**
            - Handles imbalanced data using `scale_pos_weight`
            - Captures complex interactions between features (e.g., topic + time + student history)
            - Built-in regularization prevents overfitting on our 5,800 row dataset
            - Provides feature importance rankings (shown above)
            """)
        else:
            st.markdown(f"""
            **Algorithm:** Random Forest
            
            Random Forest builds many decision trees in parallel on random subsets of data, 
            then takes a majority vote. This "bagging" approach reduces overfitting.
            
            **Why Random Forest won this round:**
            - `class_weight='balanced'` handles imbalanced data natively
            - Less prone to overfitting on our 5,800 row dataset
            - Robust to noisy features and outliers
            - Provides feature importance rankings (shown above)
            """)
        st.markdown("""
        **Key Metrics Explained:**
        - **F1 Score** - Balance between precision and recall (most important for us)
        - **AUC-ROC** - How well the model separates attendees from non-attendees
        - **Accuracy** - Overall correctness (can be misleading with imbalanced data)
        """)
        
        # retraining log
        log_path = os.path.join(MODELS_DIR, "retrain_log.txt")
        if os.path.exists(log_path):
            st.markdown("---")
            st.subheader("Retraining History")
            with open(log_path, 'r') as f:
                log_lines = f.readlines()
            if log_lines:
                for line in log_lines[-10:]:  # show last 10 entries
                    st.text(line.strip())
            else:
                st.info("No retraining attempts yet.")
    
    else:
        st.warning("No trained model found. Please run the training pipeline first.")
        st.code("python main.py", language="bash")


# ---- Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Rahul Purohit**")
st.sidebar.markdown("CSE Department")
st.sidebar.markdown(f"Data: {df['student_id'].nunique()} students, {df['event_id'].nunique()} events")
