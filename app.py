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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.predict import predict_single_event, load_latest_model
from src.feature_engineering import run_feature_pipeline

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---- Color Palette ----
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#00D2FF',
    'accent': '#FF6584',
    'success': '#00C9A7',
    'warning': '#FFB800',
    'danger': '#FF4757',
    'bg_card': '#1E1E2E',
    'bg_dark': '#0E1117',
    'text': '#E8E8E8',
    'text_dim': '#8B8FA3',
    'gradient_1': '#6C63FF',
    'gradient_2': '#00D2FF',
    'gradient_3': '#00C9A7',
    'gradient_4': '#FF6584',
}

PLOTLY_TEMPLATE = 'plotly_dark'
CHART_COLORS = ['#6C63FF', '#00D2FF', '#00C9A7', '#FF6584', '#FFB800', 
                '#A855F7', '#F472B6', '#38BDF8', '#34D399', '#FBBF24']


# ---- Page Config ----
st.set_page_config(
    page_title="Workshop Attendance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---- Custom CSS ----
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Hide default header & footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: rgba(108, 99, 255, 0.15);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2B55 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.2);
        border-color: rgba(108, 99, 255, 0.5);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #8B8FA3 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        color: #E8E8E8 !important;
        font-weight: 600 !important;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 30px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #5A52D5 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF 0%, #6C63FF 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
    }
    
    /* Selectbox / Slider */
    .stSelectbox > div > div, .stSlider > div {
        border-radius: 10px;
    }
    
    /* Dividers */
    hr { border-color: rgba(108, 99, 255, 0.15) !important; margin: 2rem 0 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF 0%, #5A52D5 100%) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(108, 99, 255, 0.08);
        border-radius: 10px;
    }
    
    /* Alert boxes */
    .stAlert { border-radius: 12px; }
    
    /* Plotly chart containers */
    .stPlotlyChart { border-radius: 16px; overflow: hidden; }
    
    /* ---- Fade-in / Lazy-load Animations ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .fade-in-section {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    .fade-in-delay-1 { animation: fadeInUp 0.6s ease-out 0.1s forwards; opacity: 0; }
    .fade-in-delay-2 { animation: fadeInUp 0.6s ease-out 0.2s forwards; opacity: 0; }
    .fade-in-delay-3 { animation: fadeInUp 0.6s ease-out 0.35s forwards; opacity: 0; }
    .fade-in-delay-4 { animation: fadeInUp 0.6s ease-out 0.5s forwards; opacity: 0; }
    .fade-in-delay-5 { animation: fadeInUp 0.6s ease-out 0.65s forwards; opacity: 0; }
    
    /* Splash screen */
    .splash-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 40%, #16213e 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 99999;
        animation: splashFadeOut 0.5s ease-in 2.2s forwards;
        pointer-events: auto;
    }
    .splash-overlay.hidden {
        pointer-events: none;
    }
    @keyframes splashFadeOut {
        to { opacity: 0; visibility: hidden; pointer-events: none; }
    }
    .splash-logo {
        font-size: 4rem;
        margin-bottom: 16px;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .splash-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .splash-sub {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        color: #8B8FA3;
        margin-bottom: 32px;
    }
    .splash-bar-track {
        width: 220px;
        height: 4px;
        background: rgba(108, 99, 255, 0.15);
        border-radius: 4px;
        overflow: hidden;
    }
    .splash-bar-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #6C63FF, #00D2FF);
        border-radius: 4px;
        animation: splashProgress 2s ease-in-out forwards;
    }
    @keyframes splashProgress {
        0% { width: 0%; }
        60% { width: 70%; }
        100% { width: 100%; }
    }
    
    /* Custom gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .subtitle { color: #8B8FA3; font-size: 1.1rem; margin-top: -10px; }
    
    /* Stat card custom HTML */
    .stat-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2B55 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-card .value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .label {
        font-size: 0.85rem;
        color: #8B8FA3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Prediction result cards */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #1E1E2E 100%);
        border-left: 4px solid;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .result-card.purple { border-color: #6C63FF; }
    .result-card.blue { border-color: #00D2FF; }
    .result-card.green { border-color: #00C9A7; }
    .result-card.pink { border-color: #FF6584; }
</style>
""", unsafe_allow_html=True)


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
    """Load metadata for the best model (highest F1) and comparison data"""
    # try comparison file first
    comp_path = os.path.join(MODELS_DIR, "model_comparison.json")
    comparison = None
    winner = 'xgboost'
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comparison = json.load(f)
        winner = comparison.get('winner', 'xgboost')
    
    # load the winner's metadata
    meta_path = os.path.join(MODELS_DIR, f"{winner}_latest_meta.json")
    if not os.path.exists(meta_path):
        # fallback: check all models
        for name in ['xgboost', 'random_forest', 'logistic_regression']:
            meta_path = os.path.join(MODELS_DIR, f"{name}_latest_meta.json")
            if os.path.exists(meta_path):
                break
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            info = json.load(f)
        info['comparison'] = comparison
        return info
    return None


# ---- Sidebar ----
with st.sidebar:
    st.markdown('<h1 style="text-align:center;">🎯</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center; margin-top:-10px;">Workshop Predictor</h3>', unsafe_allow_html=True)
    st.markdown("")
    
    page = st.radio("Navigation", [
        "🏠  Overview",
        "🔮  Predict Attendance",
        "📈  Attendance Trends",
        "🔍  Topic Analysis",
        "⚙️  Model Performance"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # sidebar footer
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: #8B8FA3; font-size: 0.8rem; margin: 4px 0;">Built by</p>
        <p style="font-weight: 600; font-size: 0.95rem; margin: 4px 0;">Rahul Purohit</p>
        <p style="color: #6C63FF; font-size: 0.8rem; margin: 4px 0;">School of Technology</p>
    </div>
    """, unsafe_allow_html=True)


# ---- Splash Screen (first load only) ----
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = True
    st.markdown("""
    <div class="splash-overlay" id="splashScreen">
        <div class="splash-logo">🎯</div>
        <div class="splash-title">Workshop Predictor</div>
        <div class="splash-sub">Loading ML-powered insights...</div>
        <div class="splash-bar-track">
            <div class="splash-bar-fill"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Load Data ----
df = load_data()
df['event_date'] = pd.to_datetime(df['event_date'])


# ---- Helper: plotly layout defaults ----
def clean_layout(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color='#E8E8E8'),
        xaxis=dict(gridcolor='rgba(108,99,255,0.08)'),
        yaxis=dict(gridcolor='rgba(108,99,255,0.08)'),
    )
    return fig


# ========================================================
# PAGE: Overview
# ========================================================
if page == "🏠  Overview":
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Workshop Attendance Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ML-powered insights to help organizers plan better events</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    
    # key metrics
    st.markdown('<div class="fade-in-delay-1">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Events", f"{df['event_id'].nunique()}")
    with c2:
        st.metric("Total Students", f"{df['student_id'].nunique()}")
    with c3:
        st.metric("Registrations", f"{len(df):,}")
    with c4:
        st.metric("Avg Attendance", f"{df['attended'].mean():.1%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # charts
    st.markdown('<div class="fade-in-delay-3">', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        topic_rates = df.groupby('topic')['attended'].agg(['mean', 'count']).sort_values('mean', ascending=True).reset_index()
        fig = go.Figure(go.Bar(
            y=topic_rates['topic'],
            x=topic_rates['mean'],
            orientation='h',
            marker=dict(
                color=topic_rates['mean'],
                colorscale=[[0, '#6C63FF'], [0.5, '#00D2FF'], [1, '#00C9A7']],
                cornerradius=6,
            ),
            text=[f"  {v:.0%}  (n={c})" for v, c in zip(topic_rates['mean'], topic_rates['count'])],
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='<b>%{y}</b><br>Rate: %{x:.1%}<br>Events: %{text}<extra></extra>',
        ))
        fig.update_layout(title=dict(text='📊 Attendance by Topic', font=dict(size=18)))
        fig.update_xaxes(title='Attendance Rate', tickformat='.0%', range=[0, max(topic_rates['mean'])*1.3])
        fig.update_yaxes(title='')
        clean_layout(fig, height=500)
        st.plotly_chart(fig, width='stretch')
    
    with col_right:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_rates = df.groupby('day_of_week')['attended'].mean().reindex(day_order).reset_index()
        day_rates.columns = ['day', 'rate']
        colors = [COLORS['danger'] if d in ['Saturday', 'Sunday'] else COLORS['primary'] for d in day_order]
        
        fig = go.Figure(go.Bar(
            x=day_rates['day'],
            y=day_rates['rate'],
            marker=dict(color=colors, cornerradius=8),
            text=[f"{v:.0%}" for v in day_rates['rate']],
            textposition='outside',
            textfont=dict(size=12, color='#E8E8E8'),
            hovertemplate='<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>',
        ))
        fig.update_layout(title=dict(text='📅 Weekday vs Weekend', font=dict(size=18)))
        fig.update_yaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_xaxes(title='')
        clean_layout(fig, height=500)
        st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # model info card
    model_info = load_model_info()
    if model_info:
        st.markdown('<div class="fade-in-delay-5">', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🤖 Active Model")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Model", model_info['model_name'].upper().replace('_', ' '))
        with m2:
            st.metric("F1 Score", f"{model_info['metrics']['f1_score']:.3f}")
        with m3:
            st.metric("AUC-ROC", f"{model_info['metrics']['auc_roc']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================================
# PAGE: Predict Attendance  
# ========================================================
elif page == "🔮  Predict Attendance":
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Predict Attendance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configure your upcoming workshop to get an AI-powered turnout prediction</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("##### 📋 Event Details")
            topic = st.selectbox("Workshop Topic", sorted(df['topic'].unique()))
            speaker_type = st.selectbox("Speaker Type", sorted(df['speaker_type'].unique()))
            day_of_week = st.selectbox("Day of Week", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            time_slot = st.selectbox("Time Slot", sorted(df['time_slot'].unique()))
            event_date = st.date_input("Event Date")
        
        with col2:
            st.markdown("##### ⚙️ Configuration")
            mode = st.selectbox("Mode", ['Offline', 'Online'])
            duration = st.slider("Duration (minutes)", 30, 240, 90, step=30)
            num_registrations = st.slider("Expected Registrations", 10, 150, 50)
            exam_proximity = st.select_slider("Exam Proximity", 
                options=[1, 2, 3], 
                format_func=lambda x: {1: "🔴 Near Exams", 2: "🟡 Moderate", 3: "🟢 Far from Exams"}[x])
            promotion_level = st.selectbox("Promotion Level", ['Low', 'Medium', 'High'])
        
        st.markdown("")
        submitted = st.form_submit_button("🎯 Predict Attendance", width="stretch")
    
    if submitted:
        event_details = {
            'topic': topic, 'speaker_type': speaker_type,
            'day_of_week': day_of_week, 'time_slot': time_slot,
            'duration_minutes': duration, 'mode': mode,
            'exam_proximity': exam_proximity, 'promotion_level': promotion_level,
            'num_registrations': num_registrations, 'event_date': str(event_date)
        }
        
        with st.spinner("🔮 Running prediction model..."):
            try:
                result = predict_single_event(event_details, df)
                
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                st.markdown("")
                
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Registered", result['total_registered'])
                with r2:
                    st.metric("Predicted Attendees", result['predicted_attendees'])
                with r3:
                    st.metric("Predicted Rate", f"{result['attendance_rate']:.1%}")
                with r4:
                    conf_map = {'High': '🟢 High', 'Medium': '🟡 Medium', 'Low': '🔴 Low'}
                    st.metric("Confidence", conf_map.get(result['confidence'], result['confidence']))
                
                # visual gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['attendance_rate'] * 100,
                    title={'text': "Predicted Attendance Rate", 'font': {'size': 16, 'color': '#E8E8E8'}},
                    number={'suffix': '%', 'font': {'size': 40, 'color': '#E8E8E8'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#8B8FA3'},
                        'bar': {'color': '#6C63FF'},
                        'bgcolor': '#1E1E2E',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(255,71,87,0.2)'},
                            {'range': [30, 60], 'color': 'rgba(255,184,0,0.2)'},
                            {'range': [60, 100], 'color': 'rgba(0,201,167,0.2)'},
                        ],
                        'threshold': {
                            'line': {'color': '#00D2FF', 'width': 3},
                            'thickness': 0.8,
                            'value': result['attendance_rate'] * 100
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=280, margin=dict(l=30, r=30, t=40, b=10),
                    font=dict(family='Plus Jakarta Sans')
                )
                st.plotly_chart(fig, width='stretch')
                
                # recommendation
                attendees = result['predicted_attendees']
                if attendees < 20:
                    st.error("📉 **Low predicted turnout.** Consider changing the topic, timing, or boosting promotion.")
                elif attendees < 40:
                    st.warning("📊 **Moderate turnout expected.** A medium-sized room should work.")
                else:
                    st.success("🚀 **Great turnout expected!** Prepare a larger venue and extra resources.")
                
                topic_history = df[df['topic'] == topic]
                if len(topic_history) > 0:
                    hist_rate = topic_history['attended'].mean()
                    st.info(f"📚 **Historical reference:** *{topic}* workshops average "
                            f"**{hist_rate:.1%}** attendance across {topic_history['event_id'].nunique()} past events.")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.code("python main.py", language="bash")


# ========================================================
# PAGE: Attendance Trends
# ========================================================
elif page == "📈  Attendance Trends":
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Attendance Trends</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover patterns and trends across time, exams, and speakers</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    
    # monthly trend
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['event_date'].dt.to_period('M').astype(str)
    monthly = df_monthly.groupby('month').agg(
        registrations=('attended', 'count'),
        attended=('attended', 'sum'),
        rate=('attended', 'mean')
    ).reset_index()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Monthly Registrations vs Attendance", "Attendance Rate"))
    
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['registrations'],
        name='Registrations', mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['attended'],
        name='Attended', mode='lines+markers',
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=8),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=monthly['month'], y=monthly['rate'],
        name='Rate', marker=dict(color=COLORS['warning'], cornerradius=6, opacity=0.85),
        text=[f"{v:.0%}" for v in monthly['rate']], textposition='outside',
    ), row=2, col=1)
    fig.add_hline(y=df['attended'].mean(), line_dash='dash', line_color=COLORS['danger'],
                  annotation_text=f"Avg: {df['attended'].mean():.0%}", row=2, col=1)
    
    fig.update_yaxes(title='Count', row=1, col=1)
    fig.update_yaxes(title='Rate', tickformat='.0%', row=2, col=1)
    fig.update_xaxes(tickangle=45)
    clean_layout(fig, height=550)
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Impact of Exam Proximity")
        exam_data = df.groupby('exam_proximity')['attended'].mean().reset_index()
        exam_data['label'] = exam_data['exam_proximity'].map({1: '🔴 Near', 2: '🟡 Moderate', 3: '🟢 Far'})
        fig = go.Figure(go.Bar(
            x=exam_data['label'], y=exam_data['attended'],
            marker=dict(color=[COLORS['danger'], COLORS['warning'], COLORS['success']], cornerradius=10),
            text=[f"{v:.1%}" for v in exam_data['attended']], textposition='outside',
            textfont=dict(size=14, color='#E8E8E8'),
        ))
        fig.update_layout(title='Students skip more near exams')
        fig.update_yaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_xaxes(title='')
        clean_layout(fig, height=350)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🎤 Impact of Speaker Type")
        speaker = df.groupby('speaker_type')['attended'].mean().sort_values().reset_index()
        fig = go.Figure(go.Bar(
            y=speaker['speaker_type'], x=speaker['attended'],
            orientation='h',
            marker=dict(
                color=speaker['attended'],
                colorscale=[[0, '#6C63FF'], [1, '#00D2FF']],
                cornerradius=10,
            ),
            text=[f"  {v:.1%}" for v in speaker['attended']], textposition='outside',
            textfont=dict(size=13),
        ))
        fig.update_layout(title='Industry speakers draw the most')
        fig.update_xaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_yaxes(title='')
        clean_layout(fig, height=350)
        st.plotly_chart(fig, width='stretch')
    
    # heatmap
    st.markdown("### 🗓️ Day × Time Slot Heatmap")
    st.info(
        "**Why Day × Time Slot?** These are the two scheduling factors organizers "
        "can directly control. Other features (speaker, topic, exam proximity) are "
        "already shown above individually. A heatmap of Day vs Time reveals their "
        "**interaction effect** — e.g., afternoons are great on weekdays but not weekends. "
        "This helps organizers pick the optimal (day, slot) combo rather than looking "
        "at each factor in isolation."
    )
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    slot_day = df.groupby(['day_of_week', 'time_slot'])['attended'].mean().unstack(fill_value=0)
    slot_day = slot_day.reindex(day_order)
    
    fig = go.Figure(go.Heatmap(
        z=slot_day.values, x=slot_day.columns.tolist(), y=slot_day.index.tolist(),
        colorscale='YlOrRd',
        text=[[f"{v:.0%}" for v in row] for row in slot_day.values],
        texttemplate="%{text}", textfont=dict(size=12, color='black'),
        hovertemplate='<b>%{y} - %{x}</b><br>Rate: %{z:.1%}<extra></extra>',
    ))
    fig.update_layout(title='When do students show up the most?')
    clean_layout(fig, height=350)
    st.plotly_chart(fig, width='stretch')


# ========================================================
# PAGE: Topic Analysis
# ========================================================
elif page == "🔍  Topic Analysis":
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Topic & Event Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep-dive into attendance patterns for any topic</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    
    selected_topic = st.selectbox("🔎 Select Topic", 
                                  ['All Topics'] + sorted(df['topic'].unique().tolist()))
    
    topic_df = df if selected_topic == 'All Topics' else df[df['topic'] == selected_topic]
    
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Events", topic_df['event_id'].nunique())
    with c2:
        st.metric("Registrations", f"{len(topic_df):,}")
    with c3:
        st.metric("Attended", f"{int(topic_df['attended'].sum()):,}")
    with c4:
        st.metric("Rate", f"{topic_df['attended'].mean():.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏛️ By Department")
        dept = topic_df.groupby('department')['attended'].mean().sort_values().reset_index()
        fig = go.Figure(go.Bar(
            y=dept['department'], x=dept['attended'], orientation='h',
            marker=dict(
                color=dept['attended'],
                colorscale=[[0, '#6C63FF'], [1, '#00C9A7']],
                cornerradius=8,
            ),
            text=[f"  {v:.1%}" for v in dept['attended']], textposition='outside',
        ))
        fig.update_xaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_yaxes(title='')
        clean_layout(fig, height=300)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🎓 By Semester")
        sem = topic_df.groupby('semester')['attended'].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=sem['semester'].astype(str), y=sem['attended'],
            marker=dict(color=COLORS['secondary'], cornerradius=8),
            text=[f"{v:.0%}" for v in sem['attended']], textposition='outside',
        ))
        fig.update_xaxes(title='Semester')
        fig.update_yaxes(title='Attendance Rate', tickformat='.0%')
        clean_layout(fig, height=300)
        st.plotly_chart(fig, width='stretch')
    
    # mode + club
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💻 Online vs Offline")
        mode_data = topic_df.groupby('mode').agg(
            count=('attended', 'count'), attended=('attended', 'sum'), rate=('attended', 'mean')
        ).reset_index()
        fig = go.Figure(go.Bar(
            x=mode_data['mode'], y=mode_data['rate'],
            marker=dict(color=[COLORS['primary'], COLORS['success']], cornerradius=10),
            text=[f"{v:.1%}<br><span style='font-size:10px'>{a}/{c}</span>" 
                  for v, a, c in zip(mode_data['rate'], mode_data['attended'], mode_data['count'])],
            textposition='outside',
        ))
        fig.update_yaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_xaxes(title='')
        clean_layout(fig, height=300)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🏃 Club Activity Impact")
        club = topic_df.groupby('club_activity_level')['attended'].mean()
        club_order = ['Low', 'Medium', 'High']
        club_vals = [club.get(l, 0) for l in club_order]
        club_colors = [COLORS['danger'], COLORS['warning'], COLORS['success']]
        
        fig = go.Figure(go.Bar(
            x=club_order, y=club_vals,
            marker=dict(color=club_colors, cornerradius=10),
            text=[f"{v:.1%}" for v in club_vals], textposition='outside',
            textfont=dict(size=14),
        ))
        fig.update_yaxes(title='Attendance Rate', tickformat='.0%')
        fig.update_xaxes(title='Club Activity Level')
        clean_layout(fig, height=300)
        st.plotly_chart(fig, width='stretch')


# ========================================================
# PAGE: Model Performance
# ========================================================
elif page == "⚙️  Model Performance":
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text">Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Details on the active ML model and its key metrics</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    
    model_info = load_model_info()
    
    if model_info:
        metrics = model_info['metrics']
        comparison = model_info.get('comparison')
        
        # model metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            winner_name = model_info['model_name'].upper().replace('_', ' ')
            st.metric("Winner Model", winner_name)
        with c2:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with c3:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        with c4:
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
        
        st.markdown("")
        
        # === Model Comparison Table ===
        if comparison:
            st.markdown("### 📊 Model Comparison")
            st.markdown("*All 3 models trained on the same data. Winner selected by F1 Score.*")
            
            comp_data = []
            winner = comparison.get('winner', '')
            for name in ['xgboost', 'random_forest', 'logistic_regression']:
                if name in comparison and isinstance(comparison[name], dict):
                    m = comparison[name]
                    display_name = name.upper().replace('_', ' ')
                    if name == winner:
                        display_name += " ⭐"
                    comp_data.append({
                        'Model': display_name,
                        'F1 Score': f"{m['f1_score']:.4f}",
                        'AUC-ROC': f"{m['auc_roc']:.4f}",
                        'Accuracy': f"{m['accuracy']:.4f}",
                        'Threshold': f"{m.get('threshold', 0.5):.2f}",
                    })
            
            if comp_data:
                st.dataframe(
                    pd.DataFrame(comp_data).set_index('Model'),
                    width="stretch"
                )
            
            # grouped bar chart comparison
            bar_data = []
            for name in ['xgboost', 'random_forest', 'logistic_regression']:
                if name in comparison and isinstance(comparison[name], dict):
                    m = comparison[name]
                    display_name = name.replace('_', ' ').title()
                    bar_data.append({'Model': display_name, 'Metric': 'F1 Score', 'Value': m['f1_score']})
                    bar_data.append({'Model': display_name, 'Metric': 'AUC-ROC', 'Value': m['auc_roc']})
                    bar_data.append({'Model': display_name, 'Metric': 'Accuracy', 'Value': m['accuracy']})
            
            if bar_data:
                bar_df = pd.DataFrame(bar_data)
                fig = px.bar(
                    bar_df, x='Metric', y='Value', color='Model', barmode='group',
                    color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent']],
                    text='Value'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside', textfont_size=11)
                fig.update_yaxes(range=[0, 1])
                fig.update_layout(title='Model Performance Comparison')
                clean_layout(fig, height=400)
                st.plotly_chart(fig, width='stretch')
            
            st.markdown("")
        
        # radar chart + threshold gauge
        col1, col2 = st.columns([1, 1])
        
        with col1:
            categories = ['Accuracy', 'F1 Score', 'AUC-ROC']
            values = [metrics['accuracy'], metrics['f1_score'], metrics['auc_roc']]
            
            fig = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(108, 99, 255, 0.2)',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, color=COLORS['secondary']),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(range=[0, 1], gridcolor='rgba(108,99,255,0.15)', tickfont=dict(size=10)),
                    angularaxis=dict(gridcolor='rgba(108,99,255,0.15)', tickfont=dict(size=13)),
                ),
                title=dict(text='📐 Metric Overview', font=dict(size=18)),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Plus Jakarta Sans', color='#E8E8E8'),
                height=350, margin=dict(l=60, r=60, t=60, b=30),
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # threshold info
            threshold = metrics.get('threshold', 0.5)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=threshold,
                title={'text': "Optimized Threshold", 'font': {'size': 16, 'color': '#E8E8E8'}},
                number={'font': {'size': 36, 'color': '#E8E8E8'}},
                gauge={
                    'axis': {'range': [0, 1], 'tickcolor': '#8B8FA3'},
                    'bar': {'color': COLORS['success']},
                    'bgcolor': '#1E1E2E',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 0.3], 'color': 'rgba(255,71,87,0.15)'},
                        {'range': [0.3, 0.6], 'color': 'rgba(255,184,0,0.15)'},
                        {'range': [0.6, 1], 'color': 'rgba(0,201,167,0.15)'},
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(l=30, r=30, t=60, b=10),
                font=dict(family='Plus Jakarta Sans'),
            )
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # feature importance
        feat_imp = load_feature_importance()
        if feat_imp is not None:
            st.markdown("### 🏆 Top 15 Most Important Features")
            top_15 = feat_imp.head(15).sort_values('importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                y=top_15['feature'], x=top_15['importance'], orientation='h',
                marker=dict(
                    color=top_15['importance'],
                    colorscale=[[0, '#6C63FF'], [0.5, '#00D2FF'], [1, '#00C9A7']],
                    cornerradius=6,
                ),
                text=[f"  {v:.4f}" for v in top_15['importance']], textposition='outside',
                textfont=dict(size=11),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
            ))
            fig.update_layout(title='What factors matter most for attendance?')
            fig.update_xaxes(title='Importance Score')
            fig.update_yaxes(title='')
            clean_layout(fig, height=450)
            st.plotly_chart(fig, width='stretch')
        
        # how it works
        st.markdown("---")
        st.markdown("### 💡 How the Model Works")
        
        with st.expander("Click to learn about the models", expanded=False):
            st.markdown(f"""
            **3 Models Trained & Compared:**
            
            | Model | Type | Key Idea |
            |-------|------|----------|
            | **XGBoost** | Gradient Boosting | Builds trees sequentially — each tree corrects mistakes from previous ones |
            | **Random Forest** | Bagging | Builds trees independently and averages them — robust but less precise |
            | **Logistic Regression** | Linear Model | Fits a linear decision boundary — baseline to prove non-linear patterns exist |
            
            **Why XGBoost usually wins on our data:**
            - `scale_pos_weight` handles class imbalance natively
            - Captures complex feature interactions (topic × time × student history)
            - Built-in L1/L2 regularization prevents overfitting on {len(df):,} rows
            - Sequential error correction beats independent averaging
            
            **The winner is selected by F1 Score** (not accuracy), because accuracy
            is misleading on imbalanced data.
            """)
            
            st.markdown("""
            ---
            **Evaluation Metrics:**
            | Metric | What it measures |
            |--------|------------------|
            | **F1 Score** | Balance between precision & recall *(primary metric)* |
            | **AUC-ROC** | How well the model separates attendees from no-shows |
            | **Precision** | Of those predicted to attend, how many actually did |
            | **Recall** | Of those who attended, how many did we predict correctly |
            | **Accuracy** | Overall correctness *(misleading with imbalanced data)* |
            """)
        
        with st.expander("🔬 Multicollinearity & Feature Strategy", expanded=False):
            st.markdown("""
            **69 features from 20 raw columns — isn't that multicollinear?**
            
            Yes, engineered features are **intentionally correlated** with their sources.
            For example, `cgpa_band_high` is derived from `cgpa`, so they will correlate.
            Here's why that's handled:
            
            | Model | Multicollinearity Impact | How We Handle It |
            |-------|------------------------|------------------|
            | **XGBoost** | ✅ Immune — tree splits don't care about correlation | L1/L2 regularization prunes redundant splits |
            | **Random Forest** | ✅ Immune — random feature sampling at each split reduces redundancy | `max_features='sqrt'` ensures diversity |
            | **Logistic Regression** | ⚠️ Affected — correlated features inflate coefficients | Used as a **baseline only**, not for interpretation |
            
            **Why 69 features anyway?**
            - Raw columns have weak correlation with `attended` (~0.08 max)
            - Interaction terms like `dept_topic_match` and `recent_att_x_event` capture **non-linear patterns** raw columns miss
            - Tree models automatically ignore unhelpful features (importance → 0)
            - The feature importance chart above shows which features actually matter
            
            **In short:** More features give the model more signal to choose from.
            Tree-based models are excellent at selecting useful ones and ignoring noise.
            """)
        
        # retraining log
        log_path = os.path.join(MODELS_DIR, "retrain_log.txt")
        if os.path.exists(log_path):
            with st.expander("📜 Retraining History"):
                with open(log_path, 'r') as f:
                    log_lines = f.readlines()
                if log_lines:
                    for line in log_lines[-10:]:
                        st.code(line.strip(), language=None)
                else:
                    st.info("No retraining attempts yet.")
    
    else:
        st.warning("⚠️ No trained model found. Run the training pipeline first:")
        st.code("python main.py", language="bash")
