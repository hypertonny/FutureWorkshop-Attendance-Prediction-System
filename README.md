<div align="center">

# 🎯 FutureWorkshop — Attendance Prediction System

**Predict. Plan. Pack the room.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

*An ML-powered system built for **Vijaybhoomi University** that predicts student turnout for FutureWorkshop events — helping organizers plan better workshops, allocate resources, and boost engagement across all four schools.*

<br>

[🚀 Quick Start](#-quick-start) · [📊 Dashboard](#-dashboard) · [🏗️ Architecture](#️-architecture) · [🧠 How It Works](#-how-it-works) · [📁 Project Structure](#-project-structure)

</div>

---

## ❓ The Problem

At **Vijaybhoomi University**, we run **FutureWorkshop** events regularly — bringing in industry professionals, alumni, and domain experts as guest speakers to share real-world knowledge with students across all four schools.

The problem is painfully simple:

> *"We invited a guest speaker who flew in from Bangalore. 80 students registered. 15 showed up. The auditorium was embarrassingly empty."*

When a guest takes time out of their schedule to come speak at our campus, they deserve to see a room full of engaged students — not rows of empty chairs. And when students *do* show up, they deserve a well-organized event, not a chaotic scramble because we overbooked.


- **Embarrassed guests** — speakers who prepared for 100 people but present to 20
- **Missed learning** — if we knew turnout would be low, we could have promoted harder or rescheduled2

With four distinct schools (Technology, Design, Business, Music) running cross-disciplinary workshops on 16 different topics, the attendance pattern isn't random — it's *predictable*. A Data Science talk will pack the room with Tech students but barely draw from Music. An industry speaker on a weekday afternoon during exams? Expect a ghost town.

**This system turns that guesswork into a data-driven prediction.**

---

## 💡 The Solution

A machine learning pipeline that learns from **historical attendance patterns** — which topics draw which schools, how speaker type affects turnout, whether exam season kills attendance — and predicts how many students will *actually* walk through the auditorium doors for a new event.

**For organizers:** Know in advance if you'll get 30 or 130 — plan seating, catering, and promotion accordingly.
**For guest speakers:** Walk into a room that's full, not half-empty. Their time and expertise deserve that respect.

### Vijaybhoomi University — 4 Schools

| School                             | Domain Topics                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| 🖥️**School of Technology** | Data Science, ML, AI & Deep Learning, Web Dev, Cybersecurity, Cloud Computing |
| 🎨**School of Design**       | UI/UX Design, Design Thinking, Branding & Identity, Creative Coding           |
| 💼**School of Business**     | Entrepreneurship, Digital Marketing, Product Management                       |
| 🎵**School of Music**        | Music Production, Sound Design                                                |

The model captures **school-topic affinity** — e.g., Technology students are more likely to attend a Data Science workshop, while Design students gravitate toward UI/UX events.

### Key Features

| Feature                                | Description                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| 🤖**3-Model Comparison**         | XGBoost + Random Forest + Logistic Regression — **auto-selects best by F1** |
| 📊**69 Engineered Features**     | From 19 raw columns → rich behavioral signals including school-topic affinity        |
| 🏫**Cross-School Intelligence**  | School-topic affinity modeling for all 4 VBU schools & 16 workshop topics             |
| 🧪**Standalone Data Generator**  | Synthesize realistic data from scratch — no CSV needed                               |
| ♻️**Auto-Retraining Pipeline** | Hot-swap models with 1% improvement gate + dynamic winner selection                   |
| 📊**Interactive Dashboard**      | Multiple pages: Predict, EDA, Model Performance, Features, Maintenance                |
| 🗄️**Scalable Database**        | SQLite now, PostgreSQL-ready (just change one line)                                   |
| ⚖️**Imbalanced Data Handling** | SMOTE + threshold optimization for real-world skew                                    |
| 🔄**Fresh-Clone Ready**          | `python main.py` auto-generates data + trains; winner auto-selected                  |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/hypertonny/FutureWorkshop-Attendance-Prediction-System.git
cd FutureWorkshop-Attendance-Prediction-System

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize DB + Train models (auto-generates data if CSV is missing)
python main.py

# 5. Launch the dashboard
streamlit run app.py
```

> No CSV file needed — `main.py` auto-generates synthetic data on a fresh clone.
> The dashboard opens at **http://localhost:8501** 🎉

### 📦 Data Source & Synthesis

This project uses **synthetically generated data** — no external download required. The script [`generate_data.py`](generate_data.py) creates realistic workshop attendance records from scratch using probability-based rules that mimic real student behavior at Vijaybhoomi University:

- **500 students** across 4 VBU schools, each with randomized CGPA, club activity, and semester
- **100 workshop events** spanning 16 cross-school topics, with varied speakers, time slots, and modes
- **~3 900 registrations** with attendance determined by 10+ realistic factors (club activity, speaker type, exam proximity, topic popularity, registration timing, etc.)

> On a fresh clone, `python main.py` calls `generate_data.py` automatically if no CSV exists — the repo is fully self-contained.

### Data Generator CLI

```bash
# Generate from scratch with custom params
python generate_data.py --students 300 --events 50 --seed 123

# Regenerate attendance for existing CSV
python generate_data.py --regenerate

# Full help
python generate_data.py --help
```

---

## 📊 Dashboard

The Streamlit dashboard has **5 interactive pages** with a branded splash screen and lazy-load animations:

| Page                            | What it does                                                         |
| ------------------------------- | -------------------------------------------------------------------- |
| 🏠**Overview**            | Key metrics, attendance by topic & day-of-week charts                |
| 🔮**Predict Attendance**  | Enter event details → get predicted turnout + confidence            |
| 📈**Attendance Trends**   | Monthly trends, exam impact, speaker & time slot analysis            |
| 🔍**Topic Analysis**      | Deep-dive into any topic — department, semester, mode breakdown     |
| ⚙️**Model Performance** | 3-model comparison table, bar chart, radar chart, feature importance |

---

## 🏗️ Architecture

### 1. Development vs Production Flows

```
DEVELOPMENT FLOW (Local Machine)
═════════════════════════════════

User runs: python main.py
    │
    ├─ generate_data.py          ← Create/regenerate dataset
    │   └─ master_dataset.csv
    │
    ├─ SQLite database init      ← Build workshop.db
    │   └─ Load CSV → normalize tables
    │
    └─ Train Models              ← 3-model competition
        ├─ XGBoost
        ├─ Random Forest
        └─ Logistic Regression
            │
            └─ Compare F1 → Pick winner
                └─ Save to models/*.pkl + models/*.json
                
Output: Trained models ready for prediction


PRODUCTION FLOW (Docker Container)
═══════════════════════════════════

User makes API request to deployed system
    │
    ├─ Nginx listens on :80
    │   └─ Reverse proxy to api_server:8000
    │
    ├─ api_server.py               ← Main engine
    │   │
    │   ├─ Check if data exists
    │   │   └─ If missing: auto-generate via generate_data.py
    │   │
    │   ├─ Load frontend/index.html (GET /)
    │   │
    │   ├─ Serve API endpoints
    │   │   ├─ /api/health
    │   │   ├─ /api/options
    │   │   ├─ /api/charts
    │   │   ├─ /api/predict ← MAIN PREDICTION ENDPOINT
    │   │   ├─ /api/topic-analysis
    │   │   └─ /api/model-details
    │   │
    │   └─ Load ONE winner model from models/
    │       └─ Use its threshold + features for predictions
    │
    └─ Return predictions to frontend
        └─ Display attendance forecast + confidence
```

### 2. System Architecture (Current Production)

```
┌────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                │
│  Browser: frontend/index.html + script.js + styles.css            │
│  (Vanilla JS, Plotly charts, Fetch API)                           │
└────────────────────┬─────────────────────────────────────────────┘
                     │ HTTP requests (Predict, Analytics, Model Info)
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                      REVERSE PROXY LAYER                           │
│                         Nginx (Port 80)                            │
│  • Route requests to api_server:8000                              │
│  • Handle SSL/TLS (Cloudflare proxy)                              │
│  • Load balancing (future)                                         │
└────────────────────┬─────────────────────────────────────────────┘
                     │ Proxy Pass
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                    API SERVER LAYER                                │
│                  api_server.py (Port 8000)                         │
│  • HTTP request handler                                            │
│  • API endpoint logic                                              │
│  • Static frontend serving                                         │
└────┬───────────────┬──────────────────┬───────────────┬───────────┘
     │               │                  │               │
     ▼               ▼                  ▼               ▼
┌─────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Frontend│  │   Data       │  │  Model       │  │  Feature     │
│ Assets  │  │  Loading     │  │  Prediction  │  │  Engineering │
│         │  │              │  │              │  │              │
│ • HTML  │  │ • CSV check  │  │ • Load .pkl  │  │ Transform    │
│ • CSS   │  │ • Auto-gen   │  │ • Apply      │  │ user input   │
│ • JS    │  │   if missing │  │   threshold  │  │ to 69 feats  │
└─────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                     │                  │               │
                     └──────────┬───────┴───────────────┘
                                │
                     ┌──────────▼───────────┐
                     │   Prediction Engine  │
                     │   (src/predict.py)   │
                     │                      │
                     │  1. Load winner      │
                     │     model            │
                     │  2. Engineer input   │
                     │     features         │
                     │  3. Get probability  │
                     │  4. Calculate        │
                     │     confidence       │
                     │  5. Format response  │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │   JSON Response     │
                     │                     │
                     │ {                  │
                     │   attendance_rate  │
                     │   confidence       │
                     │   expected_count   │
                     │   top_factors      │
                     │   recommendation   │
                     │ }                  │
                     └────────────────────┘
```

### 3. Data & Model Training Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                               │
│                  (Run: python main.py)                           │
└──────────────────────────────────────────────────────────────────┘

         ┌─────────────────────────┐
         │  generate_data.py       │
         │                         │
         │ Synthesize realistic:  │
         │ • 500 students        │
         │ • 100 events          │
         │ • 3,900 registrations │
         │ • Attendance labels   │
         └────────────┬──────────┘
                      │ Creates
                      ▼
         ┌─────────────────────────┐
         │  master_dataset.csv     │
         │ (19 raw columns)        │
         └────────────┬──────────┘
                      │ Reads
                      ▼
         ┌─────────────────────────────────┐
         │   main.py + database.py        │
         │                                 │
         │ Normalize to SQLite:           │
         │ • students (500 rows)          │
         │ • events (100 rows)            │
         │ • registrations (3,900 rows)  │
         │ • model_versions (tracking)   │
         └────────────┬──────────────────┘
                      │ SQL JOINs
                      ▼
         ┌──────────────────────────────────┐
         │  Feature Engineering             │
         │  (src/feature_engineering.py)    │
         │                                  │
         │  19 raw columns → 69 features:  │
         │  ├─ Temporal (7)                │
         │  ├─ Student behavioral (6)      │
         │  ├─ Event popularity (4)        │
         │  ├─ School-topic affinity (64)  │
         │  └─ Interaction effects (3)     │
         │                                  │
         │  Output: X_train (3120×69)      │
         │          y_train (3120,)        │
         └────────────┬────────────────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │XGBoost  │  │Random   │  │ Logistic    │
   │         │  │Forest   │  │ Regression  │
   │ ├─SMOTE │  │ ├─SMOTE │  │ ├─SMOTE    │
   │ ├─5FCV  │  │ ├─5FCV  │  │ ├─5FCV     │
   │ ├─Thresh│  │ ├─Thresh│  │ ├─Thresh   │
   │ └─sweep │  │ └─sweep │  │ └─sweep    │
   └────┬────┘  └────┬────┘  └──────┬─────┘
        │ F1:0.7125  │ F1:0.7322  │ F1:0.7337
        └────────────┼──────────────┘
                     │ Compare F1 scores
                     ▼
        ┌─────────────────────────┐
        │  WINNER SELECTED        │
        │ Logistic Regression     │
        │ F1: 0.7337              │
        │ Threshold: 0.42         │
        │ AUC: 0.8267             │
        └────────┬────────────────┘
                 │ Save
                 ▼
        ┌──────────────────────────┐
        │  models/              │
        │  ├─ logistic_         │
        │  │  regression_       │
        │  │  latest.pkl        │
        │  └─ logistic_         │
        │     regression_       │
        │     latest_meta.json  │
        └──────────────────────────┘
```

### 4. Prediction Request Flow (Runtime)

```
User Action: Click "Predict Attendance"
    │
    └─ Frontend (script.js)
        │
        ├─ Collect event details:
        │   ├─ topic
        │   ├─ speaker_type
        │   ├─ mode (online/offline)
        │   ├─ promotion_level
        │   └─ num_registrations
        │
        └─ POST /api/predict
              │ (Sends JSON payload)
              ▼
            api_server.py (do_POST handler)
                │
                ├─ Parse request JSON
                │
                └─ Call src/predict.py:
                      predict_single_event(params)
                        │
                        ├─ Load winner model
                        │   (e.g., logistic_regression_latest.pkl)
                        │
                        ├─ Load feature metadata
                        │   (feature names, scaler)
                        │
                        ├─ Engineer input features
                        │   (19 raw → 69 features)
                        │   Using: student history, temporal context,
                        │           topic affinity, etc.
                        │
                        ├─ Get model prediction
                        │   probability = model.predict_proba([features])
                        │   # Returns 0.0–1.0 (e.g., 0.72)
                        │
                        ├─ Calculate confidence
                        │   # How certain is the model?
                        │   # Based on: probability margin, 
                        │   #            feature variance,
                        │   #            historical accuracy
                        │
                        └─ Format response:
                           {
                             "attendance_rate": 0.72,
                             "expected_students": 86,
                             "confidence": 0.89,
                             "model_used": "logistic_regression",
                             "threshold": 0.42,
                             "top_factors": [
                               {"name": "semester", "impact": "+0.15"},
                               {"name": "cgpa_avg", "impact": "+0.12"},
                               ...
                             ],
                             "recommendation": "Book large hall"
                           }
              │
              ▼
        Return JSON to frontend
            │
            ▼
        Display results:
        ├─ Prediction gauge (0–100%)
        ├─ Expected student count
        ├─ Confidence meter
        ├─ Top driving factors
        └─ Planning recommendation
```

### 5. Model Retraining & Swapping Cycle

```
Monthly Trigger: python src/retrain.py --from-db
    │
    ├─ Load latest data from SQLite database
    │
    ├─ Re-engineer all 69 features
    │
    ├─ Train 3 models (parallel):
    │   ├─ XGBoost (existing code)
    │   ├─ Random Forest
    │   └─ Logistic Regression
    │
    ├─ Evaluate each by F1 score on test set
    │   ├─ New XGBoost F1: 0.7401
    │   ├─ New RF F1: 0.7388
    │   └─ New LR F1: 0.7340
    │
    ├─ Pick NEW WINNER by highest F1:
    │   └─ XGBoost wins (0.7401)
    │
    ├─ Check improvement gate:
    │   ├─ Current champion: LR (F1: 0.7337)
    │   ├─ New winner: XGBoost (F1: 0.7401)
    │   ├─ Improvement: +0.64% (exceeds 1% gate? NO)
    │   └─ Decision: KEEP LR
    │
    └─ OR (if improvement ≥ 1%):
        ├─ Backup old model
        ├─ Deploy new winner
        ├─ Update models/*_latest.pkl
        └─ Log model version change

Notes: 
• --force flag skips the 1% gate
• All model versions stored in ModelVersion table
• Automatic safety: bad models can't break production
```

### 6. Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                   INPUT & OUTPUT SUMMARY                         │
└──────────────────────────────────────────────────────────────────┘

TRAINING INPUT:
  generate_data.py
    ↓
  master_dataset.csv
    • 500 students × 16 attributes
    • 100 events × 8 attributes
    • 3,900 registrations with attendance labels (0/1)

TRAINING PROCESS:
  Feature Engineering: 19 raw → 69 engineered features
  Model Training: 3 models in competition
  Winner Selection: Best F1 score

TRAINING OUTPUT:
  models/logistic_regression_latest.pkl
    • Trained model weights
  models/logistic_regression_latest_meta.json
    • Threshold (0.42)
    • Feature names (69)
    • Scaler params
    • Performance metrics (F1, AUC, etc.)

PREDICTION INPUT:
  Event parameters (JSON POST):
    {
      "topic": "Data Science",
      "speaker_type": "industry",
      "registered_count": 120,
      "day_of_week": "Wednesday",
      ...
    }

PREDICTION OUTPUT:
  {
    "attendance_rate": 0.72,          ← Probability (0.0–1.0)
    "expected_students": 86,          ← Registered × rate
    "confidence": 0.89,                ← Model certainty (0.0–1.0)
    "model_used": "logistic_regression",
    "threshold_used": 0.42,
    "top_factors": [...],              ← Feature importance
    "recommendation": "Book large hall"
  }
```

---

## 🛠️ Technical Stack

### Backend & ML
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.12 | Core implementation |
| **ML Models** | XGBoost, Random Forest, Logistic Regression | Ensemble + majority vote |
| **Data Processing** | Pandas, NumPy | ETL + feature engineering |
| **Imbalance Handling** | SMOTE (imlearn) | Balance skewed attendance data |
| **Model Selection** | Scikit-learn | Train/test split, CV, metrics |
| **Serialization** | Joblib | Save/load trained models |
| **Database** | SQLite (SQLAlchemy ORM) | Persistent data storage, production-ready for PostgreSQL |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Markup** | HTML5 | Static structure |
| **Styling** | CSS3 (custom dark theme) | Responsive UI with animations |
| **JavaScript** | Vanilla JS (ES6+) | Interactive tabs, API calls, charts |
| **Charts** | Plotly.js | Real-time visualizations |
| **Communication** | Fetch API | REST calls to backend |

### Deployment & Infrastructure
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Containerization** | Docker | Reproducible environments |
| **Orchestration** | Docker Compose | Multi-container coordination |
| **Web Server** | Nginx (Alpine) | Reverse proxy, HTTP/HTTPS routing |
| **API Server** | Python HTTP (SimpleHTTPServer + ThreadingHTTPServer) | Lightweight REST API + static file serving |
| **SSL/TLS** | Cloudflare SSL Proxy (or manual) | HTTPS encryption |
| **Deployment** | Cloudflare Pages + Bash scripts | Automated CI/CD |

---

## 📋 Technical Specifics

### Feature Engineering Pipeline (19 raw → 69 features)

**Temporal Features**
- `semester_week`, `days_to_exam`, `is_weekend`, `is_holiday`
- `month`, `day_of_week`, `is_exam_period`

**Student Behavioral**
- `rolling_3_attendance`, `attendance_streak`, `recent_3_rate`
- `cgpa_normalized`, `club_activity_level`, `semester_encoded`

**Event Popularity**
- `topic_popularity`, `speaker_pull` (industry vs. faculty), `dept_engagement`
- `promotion_level_encoded`, `mode_encoded` (online/offline)

**School-Topic Affinity** ⭐
- `dept_topic_match`: 4 schools × 16 topics → learned correlation matrix
- Example: Tech dept + ML topic = 0.92 affinity (high), Music dept + ML topic = 0.15 affinity (low)

**Interaction Effects**
- `combined_quality_attract` = speaker_quality × promotion × day_of_week
- `exam_pressure` = proximity_to_exam × course_importance
- `registration_commitment` = (time_since_registration / days_before_event)

### Top Engineered Features by Importance

After training, the model ranks features by their contribution to predictions. **Your current top 11:**

| Rank | Feature | Category | Impact |
|------|---------|----------|--------|
| 1 | `semester` | Temporal | Timing within academic semester matters most |
| 2 | `cgpa` | Student Behavioral | Student ability/engagement predicts attendance |
| 3 | `past_attendance_rate` | Student Behavioral | Historical attendance is best predictor |
| 4 | `past_events_count` | Student Behavioral | Students with event history more predictable |
| 5 | `duration_minutes` | Event Property | Workshop length affects turnout |
| 6 | `exam_proximity` | Temporal | Proximity to exams kills attendance |
| 7 | `num_registrations` | Event Property | Total registrants influences individual decisions |
| 8 | `month` | Temporal | Seasonal patterns emerge |
| 9 | `is_weekend` | Temporal | Weekend attendance differs from weekdays |
| 10 | `semester_week` | Temporal | Which week of semester matters |
| 11 | `student_rolling_attendance` | Student Behavioral | Recent trend predicts future action |

**Key Insight:** The **top 3 features account for ~40% of model decisions**:
1. Semester timing (when in academic year)
2. Student CGPA (academic engagement proxy)
3. Past attendance rate (behavioral inertia)

This means: **If you know when, who, and their track record — you can predict attendance pretty well.**

### Model Training Configuration

**Input**
- 500 synthetic students, 100 events, ~3,900 registrations
- 69 engineered features, 1 target (attended: 0/1)
- **Class imbalance:** ~30% attended, 70% no-show

**SMOTE Applied** when minority class < 35%
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**Threshold Optimization** (not default 0.5)
- Sweep: 0.10 → 0.60 by 0.01 increments
- Metric: F1-score (weighted harmonic mean of precision & recall)
- Default pick: threshold maximizing F1

**Cross-Validation**
- Strategy: 5-fold stratified
- Ensures minority class balanced across all folds

**Model Comparison**
```
┌─────────────────────────────────────────┐
│ Model       │ F1-Score │ AUC-ROC │ Best?
├─────────────────────────────────────────┤
│ XGBoost     │ 0.751    │ 0.805   │ ✓    
│ Random For. │ 0.698    │ 0.762   │      
│ Log. Regr.  │ 0.634    │ 0.701   │      
└─────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Response | Purpose |
|----------|--------|----------|---------|
| `/api/health` | GET | `{"ok": True, "status": "healthy"}` | Health check |
| `/api/overview` | GET | Dataset stats, model info | Dashboard summary |
| `/api/options` | GET | Dropdown lists (topics, days, etc.) | Form population |
| `/api/charts` | GET | Aggregated attendance by topic/day/school | EDA charts |
| `/api/predict` | POST | `{"predicted": 0.78, "confidence": 0.92, ...}` | **Main prediction** |
| `/api/topic-analysis` | GET | School breakdown for topic | Deep dive |
| `/api/model-details` | GET | Comparison table, feature importance | Model inspection |

---

### 🔮 Attendance Prediction: From Probability to Student Count

Your model outputs a **probability** (0.0 to 1.0), not a direct count. Here's how to convert:

**Example: Data Science Workshop**
- **Registered students:** 120
- **Model predicts:** 0.68 attendance probability
- **Expected attendees:** 120 × 0.68 = **~82 students**
- **Confidence:** 0.89 (89% sure about this prediction)

**Planning Guide Based on Predicted Attendance Rate:**

```
Predicted Rate → Expected Students → Recommendation
────────────────────────────────────────────────────
  > 0.75       → High attendance    → Book large hall
                                      Full catering
                                      Promote to invite more
              
  0.50–0.75    → Medium attendance  → Book medium hall
                                      Standard catering
                                      Monitor registrations

  < 0.50       → Low attendance     → Small, intimate setting
                                      Minimal catering
                                      Contact registrants
                                      Offer incentives
```

**Why This Matters:**

| Scenario | Registration | ML Prediction | Expected | Action |
|----------|--------------|---------------|----------|--------|
| Data Science (Tech mag) | 120 | 0.78 | 94 students | Large hall, lots of food |
| Music Production | 45 | 0.42 | 19 students | Bookshelf nook, coffee |
| Design (end of sem) | 60 | 0.55 | 33 students | Medium room, standard setup |

**Confidence Intervals:**

The `confidence` score (0.0–1.0) tells you how certain the model is:
- **Confidence > 0.85:** Trust the prediction strongly
- **Confidence 0.70–0.85:** Reasonable confidence, plan accordingly
- **Confidence < 0.70:** High uncertainty, use as guide + contact registrants

**Example API Response:**

```json
{
  "event": {
    "topic": "Data Science",
    "registered_count": 120,
    "speaker_type": "industry",
    "day_of_week": "Wednesday"
  },
  "prediction": {
    "attendance_rate": 0.72,
    "expected_students": 86,
    "confidence": 0.89,
    "model_used": "xgboost",
    "threshold_used": 0.42
  },
  "recommendation": {
    "hall_size": "large (100+ capacity)",
    "catering_headcount": 90,
    "confidence_level": "High — proceed with plan",
    "top_factors": [
      {"factor": "Data Science popularity", "impact": "+15% boost"},
      {"factor": "Industry speaker", "impact": "+12% boost"},
      {"factor": "Wednesday attendance high", "impact": "+8% boost"},
      {"factor": "Not near exams", "impact": "+5% boost"}
    ]
  }
}
```

---

**Example Prediction Payload**
```json
{
  "event_info": {
    "topic": "Data Science",
    "speaker_type": "industry",
    "mode": "offline",
    "promotion_level": "high",
    "registered_count": 45
  },
  "context": {
    "days_to_exam": 7,
    "semester_week": 12,
    "day_of_week": "Wednesday"
  }
}
```

**Example Response**
```json
{
  "predicted_attendance": 32,
  "predicted_rate": 0.71,
  "confidence": 0.89,
  "model_used": "xgboost",
  "feature_importance_top_5": [
    {"feature": "dept_topic_match", "importance": 0.187},
    {"feature": "promotion_level", "importance": 0.145},
    ...
  ]
}
```

### Database Schema (SQLite)

```sql
Students
├─ student_id (PK)
├─ school (design, tech, business, music)
├─ cgpa
└─ club_activity_level

Events
├─ event_id (PK)
├─ topic (16 categories)
├─ speaker_type (industry, faculty, alumni)
├─ event_date
└─ promotion_level

Registrations
├─ registration_id (PK)
├─ student_id (FK)
├─ event_id (FK)
├─ registered_date
└─ attended (0/1 — target)

ModelVersion
├─ model_id (PK)
├─ model_type (xgboost, random_forest, logistic_regression)
├─ timestamp
├─ f1_score
├─ threshold
└─ is_active
```

### Performance Metrics

**Why F1 Over Accuracy?**

Your attendance target is **imbalanced**: 
- ~70% students don't attend (negative class)
- ~30% students attend (positive class)

If a model always predicted "no one attends," it would have 70% accuracy — but would be worthless for planning. **F1 catches this** by balancing:

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precision** = "Of attendees I predicted, how many actually came?" (avoid over-booking)
- **Recall** = "Of students who actually attended, how many did I predict?" (don't miss attendees)

### How the Winner Model is Selected

After training all 3 models, they're ranked by **F1 score**. The highest F1 wins and gets deployed.

**Example from your running system:**

| Model | F1 Score | Accuracy | AUC-ROC | Winner? |
|-------|----------|----------|---------|---------|
| XGBoost | 0.7125 | 0.6992 | 0.7847 | ❌ |
| Random Forest | 0.7322 | 0.7095 | 0.8073 | ❌ |
| **Logistic Regression** | **0.7337** | **0.7275** | **0.8267** | ✅ |

**Why Logistic Regression won:**
- Highest F1 score (primary selection criterion)
- Excellent AUC-ROC (0.8267 = great discrimination)
- Fastest inference (important for real-time API)
- Better calibrated probabilities (threshold 0.42 is optimal)

**Note:** The winner model can change on next retraining if data shifts. This is intentional — you always get the best model for your current data.

**Test Performance by School** (from winner model)
| School | Recall | Precision | F1 | Interpretation |
|--------|--------|-----------|-----|-----------------|
| Technology | 0.72 | 0.81 | 0.76 | Catch 72% of Tech attendees, 81% of predictions correct |
| Design | 0.68 | 0.78 | 0.72 | Catch 68% of Design attendees, 78% precision |
| Business | 0.64 | 0.76 | 0.69 | Catch 64% of Business attendees, 76% precision |
| Music | 0.58 | 0.72 | 0.64 | Catch 58% of Music attendees, 72% precision |

*Lower performance for Music indicates challenging minority-within-minority dynamics.*

**When to Trust the Predictions:**
- ✅ When F1 > 0.70: Good balance of recall and precision
- ⚠️ When F1 ~ 0.60: Use with caution, higher uncertainty
- ❌ When F1 < 0.50: Model is not confident enough

### Deployment Checklist

✅ **Development**
- [ ] Run `python main.py` locally (generates data, trains models)
- [ ] Test via `python api_server.py --port 8765`
- [ ] Smoke-test `/api/predict` with sample payload

✅ **Production (Docker)**
- [ ] Build: `docker compose build`
- [ ] Deploy: `docker compose up -d` or `./deploy.sh`
- [ ] Verify: `curl http://localhost/api/health`
- [ ] Monitor: `docker compose logs -f`

✅ **Data Refresh**
- [ ] Quarterly: `python generate_data.py --regenerate`
- [ ] Monthly: `python src/retrain.py --from-db`
- [ ] Manual: `python src/retrain.py --force`

### How Training Works

Raw data has weak correlations (~0.08). The pipeline creates **5 categories** of 69 derived features — see [Feature Engineering Pipeline](#feature-engineering-pipeline-19-raw--69-features) for details.

Then three models compete **simultaneously**:

```
┌─────────────────────────────────────────────┐
│  Train 3 Models in Parallel:                │
│  • XGBoost (gradient boosting)              │
│  • Random Forest (ensemble)                 │
│  • Logistic Regression (linear baseline)   │
└─────────────────────────────────────────────┘
   ↓
SMOTE (balance if minority <35%)
   ↓
5-Fold CV + Threshold Sweep (0.10–0.60 for each model)
   ↓
┌─────────────────────────────────────────────┐
│  Compare by F1 Score:                       │
│  Model A F1: 0.7125  ❌                     │
│  Model B F1: 0.7322  ❌                     │
│  Model C F1: 0.7337  ✅ WINNER              │
└─────────────────────────────────────────────┘
   ↓
Deploy winner (only if ≥1% improvement over current)
```

**What Actually Matters:**

The **winner model changes** each time you retrain — it's determined by whichever model shows the highest F1 score on that specific dataset:

| Training Run | Winner | F1 Score | Why |
|--------------|--------|----------|-----|
| Run 1 (initial) | Logistic Regression | 0.7337 | Best F1 on this data |
| Run 2 (next month) | XGBoost | 0.7421 | ~1% better, gets promoted |
| Run 3 (new semester) | Random Forest | 0.7458 | ~0.5% better, gets promoted |

**The key insight:** You don't care if XGBoost or Logistic Regression wins — you care that **the best model for your current data gets deployed**. The selection mechanism ensures you're always using the optimal model.

---

## 📁 Project Structure

### Essential Files (Required for Runtime)

```
✅ ESSENTIAL
├── main.py                         # Entry point: data gen → DB init → training
├── api_server.py                   # Production API + frontend server (Docker runs this)
├── generate_data.py                # Data synthesis (called by main.py & api_server.py)
├── requirements.txt                # Python dependencies
│
├── src/                            # Core ML pipeline
│   ├── __init__.py
│   ├── database.py                 # SQLAlchemy ORM + CSV ↔ DB sync
│   ├── feature_engineering.py      # 19 raw columns → 69 features
│   ├── train_model.py              # 3-model training + selection
│   ├── retrain.py                  # Hot-swap retraining pipeline
│   └── predict.py                  # Single event prediction engine
│
└── frontend/                       # Production dashboard UI
    ├── index.html                  # HTML structure (3 tabs)
    ├── script.js                   # Client-side logic + API calls
    └── styles.css                  # Dark theme + responsive layout
```

### Generated / Deployment Files

```
⚠️ AUTO-GENERATED (every run of main.py)
├── data/
│   └── workshop.db                 # SQLite database (students, events, registrations)
├── models/
│   ├── xgboost_latest.pkl          # Trained model (joblib)
│   ├── logistic_regression_latest_meta.json   # Metadata + threshold
│   └── model_comparison.json       # 3-model comparison metrics
└── master_dataset.csv              # Synthetic data (3,900 registrations)

📦 INFRASTRUCTURE (Docker/deployment)
├── Dockerfile                      # Container definition (runs api_server.py)
├── docker-compose.yml              # Multi-container orchestration
├── deploy.sh                       # Automated deployment script
└── nginx/
    ├── nginx.conf                  # Web server config
    └── conf.d/app.conf             # Reverse proxy rules
```

### Optional / Documentation

```
📄 OPTIONAL
├── README.md                       # This file (docs only)
├── .gitignore                      # Git config (docs only)
├── DEPLOYMENT.md                   # Deployment guide
├── DOCKER-COMMANDS.md              # Docker cheat sheet
└── SETUP-SUMMARY.md                # Initial setup notes
```

> **Fresh clone?** Just run `python main.py` — it auto-generates data, builds the SQLite DB, and trains all models.
> **Why no master_dataset.csv in git?** It's synthesized on the fly; no need to commit generated files.

---

## � Future Roadmap

- [ ] Integrate with college LMS / Google Forms for real data
- [ ] Student-level prediction (which specific students will attend)
- [ ] Email/notification system for low predicted turnout
- [ ] Deploy on cloud with scheduled retraining
- [ ] A/B testing for promotion strategies
- [ ] Add weather data for offline event predictions
- [ ] CGPA integration from university records

---

## 🗓️ Updation & Maintenance Timelines

| Phase                             | Frequency              | Trigger                         | Action                                     |
| --------------------------------- | ---------------------- | ------------------------------- | ------------------------------------------ |
| **🔄 Model Retraining**     | Every semester start   | New semester (Aug / Jan)        | `python src/retrain.py`                  |
| **📊 Data Refresh**         | After every 10+ events | New attendance logged           | `python src/retrain.py --from-db`        |
| **🔍 Performance Audit**    | Monthly                | Accuracy drops below threshold  | Review features + threshold sweep          |
| **🧹 Data Cleanup**         | End of each semester   | Semester ends                   | Archive old data, regenerate baseline      |
| **🚀 Feature Updates**      | As needed              | New data sources (LMS, weather) | Update `feature_engineering.py`, retrain |
| **🛡️ Dependency Updates** | Quarterly              | Security patches / new releases | Update `requirements.txt`, test pipeline |

**Retraining safeguard:** The retrain pipeline only deploys a new model if it beats the current one by **≥ 1 % F1 score**, preventing unnecessary swaps from random variance.

---

<div align="center">

### Built with ☕ by Rahul Purohit

*Reg: 2024SEPVUGP0079 · VSST — Vijaybhoomi University*

<br>

⭐ **Star this repo if you found it useful!** ⭐

</div>
