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

**We can't over-book our auditorium** (fire safety, seating limits), but we also can't afford to under-prepare. Without a reliable prediction of actual turnout, organizers are stuck guessing — and that leads to:

- **Wasted resources** — catering, printed materials, AV setup for a crowd that never comes
- **Embarrassed guests** — speakers who prepared for 100 people but present to 20
- **Missed learning** — if we knew turnout would be low, we could have promoted harder or rescheduled
- **Over-booking risk** — accepting 200 registrations for a 120-seat auditorium because "only half will come" is a gamble

With four distinct schools (Technology, Design, Business, Music) running cross-disciplinary workshops on 16 different topics, the attendance pattern isn't random — it's *predictable*. A Data Science talk will pack the room with Tech students but barely draw from Music. An industry speaker on a weekday afternoon during exams? Expect a ghost town.

**This system turns that guesswork into a data-driven prediction.**

---

## 💡 The Solution

A machine learning pipeline that learns from **historical attendance patterns** — which topics draw which schools, how speaker type affects turnout, whether exam season kills attendance — and predicts how many students will *actually* walk through the auditorium doors for a new event.

**For organizers:** Know in advance if you'll get 30 or 130 — plan seating, catering, and promotion accordingly.  
**For guest speakers:** Walk into a room that's full, not half-empty. Their time and expertise deserve that respect.

### Vijaybhoomi University — 4 Schools

| School | Domain Topics |
|--------|--------------|
| 🖥️ **School of Technology** | Data Science, ML, AI & Deep Learning, Web Dev, Cybersecurity, Cloud Computing |
| 🎨 **School of Design** | UI/UX Design, Design Thinking, Branding & Identity, Creative Coding |
| 💼 **School of Business** | Entrepreneurship, Digital Marketing, Product Management |
| 🎵 **School of Music** | Music Production, Sound Design |

The model captures **school-topic affinity** — e.g., Technology students are more likely to attend a Data Science workshop, while Design students gravitate toward UI/UX events.

### Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **3-Model Comparison** | XGBoost + Random Forest + Logistic Regression — automatically picks the winner by F1 |
| 📊 **69 Engineered Features** | From 19 raw columns → rich behavioral signals including school-topic affinity |
| 🏫 **Cross-School Intelligence** | School-topic affinity modeling for all 4 VBU schools & 16 workshop topics |
| 🧪 **Standalone Data Generator** | Synthesize realistic data from scratch — no CSV needed |
| ♻️ **Auto-Retraining Pipeline** | Hot-swap models with 1% improvement gate |
| 📊 **Interactive Dashboard** | 5-page Streamlit app with predictions, analytics & splash screen |
| 🗄️ **Scalable Database** | SQLite now, PostgreSQL-ready (just change one line) |
| ⚖️ **Imbalanced Data Handling** | SMOTE + threshold optimization for real-world skew |
| 🔄 **Fresh-Clone Ready** | `python main.py` auto-generates data if CSV is missing |

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

This project uses **synthetically generated data** — no external download required.  
The script [`generate_data.py`](generate_data.py) creates realistic workshop attendance records from scratch using probability-based rules that mimic real student behavior at Vijaybhoomi University:

- **500 students** across 4 VBU schools, each with randomized CGPA, club activity, and semester
- **100 workshop events** spanning 16 cross-school topics, with varied speakers, time slots, and modes
- **~8 000 registrations** with attendance determined by 10+ realistic factors (club activity, speaker type, exam proximity, topic popularity, registration timing, etc.)

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

| Page | What it does |
|------|-------------|
| 🏠 **Overview** | Key metrics, attendance by topic & day-of-week charts |
| 🔮 **Predict Attendance** | Enter event details → get predicted turnout + confidence |
| 📈 **Attendance Trends** | Monthly trends, exam impact, speaker & time slot analysis |
| 🔍 **Topic Analysis** | Deep-dive into any topic — department, semester, mode breakdown |
| ⚙️ **Model Performance** | 3-model comparison table, bar chart, radar chart, feature importance |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  master_dataset  │────▶│  Feature Engine   │────▶│   Model Training    │
│     .csv         │     │  (19 → 69 feat)  │     │  XGB + RF + LR      │
└─────────────────┘     └──────────────────┘     └─────────┬───────────┘
        │                        │                         │
        │              school-topic affinity                │
        │              16 topics × 4 schools                │
        │                                                   │
        ┌──────────────────────────────────────────────────┘
        ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  SQLite DB   │     │  Best Model .pkl │     │  Streamlit App   │
│  (normalized)│     │  + metadata.json │────▶│  (5 pages)       │
└──────────────┘     └──────────────────┘     └──────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Retrain Pipeline│
                     │  (hot-swap)      │
                     └─────────────────┘
```

---

## 🧠 How It Works

### Feature Engineering Pipeline

Raw data has weak correlations (~0.08). The pipeline creates **5 categories** of derived features:

| Category | Examples | Why it helps |
|----------|----------|-------------|
| ⏰ **Temporal** | `semester_week`, `is_weekend`, `month` | Attendance drops late in semester |
| 👤 **Student History** | `rolling_attendance`, `streak`, `recent_3_rate` | Past behavior predicts future |
| 🔥 **Event Popularity** | `topic_popularity`, `speaker_pull`, `dept_engagement` | Some topics just hit different |
| 🏫 **School-Topic Affinity** | `dept_topic_match` | Tech students → Data Science, Design students → UI/UX |
| 🔗 **Interactions** | `combined_quality_attract`, `exam_pressure`, `registration_commitment` | Combined effects matter |

### Model Training

```
Raw Data → NaN Imputation (median) → SMOTE (if imbalanced)
    → Train XGBoost + Random Forest + Logistic Regression
    → Threshold Sweep (0.10 → 0.60)
    → Compare all 3 by F1 → Save winner
```

- **3 Models**: XGBoost (gradient boosting), Random Forest (bagging), Logistic Regression (linear baseline with StandardScaler)
- **NaN Handling**: Remaining NaN filled with column medians for LR/RF compatibility
- **SMOTE**: Only applied when minority class < 35%
- **Threshold Optimization**: Sweeps 0.10–0.60, picks threshold that maximizes F1
- **5-Fold Cross Validation**: Ensures scores aren't just lucky splits
- **Winner Selection**: Best F1 score wins, all 3 models saved for comparison

### Retraining Pipeline

```bash
python src/retrain.py              # retrain from CSV
python src/retrain.py --from-db    # retrain from database
python src/retrain.py --force      # force deploy regardless
```

The pipeline only promotes a new model if it beats the current one by **≥ 1% F1** — preventing unnecessary swaps from random variance.

---

## 📁 Project Structure

```
├── main.py                    # Entry point: auto-generate data → init DB → train
├── app.py                     # Streamlit dashboard (5 pages, splash screen)
├── requirements.txt           # Dependencies
├── generate_data.py           # Standalone data generator (CLI + programmatic)
├── master_dataset.csv         # Training data (auto-generated if missing)
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLAlchemy ORM (4 tables)
│   ├── feature_engineering.py # 19 raw → 69 features (incl. school-topic affinity)
│   ├── train_model.py         # XGBoost + RF + LR training + NaN imputation
│   ├── retrain.py             # Hot-retraining pipeline (1% improvement gate)
│   └── predict.py             # Prediction engine (handles missing columns)
│
├── models/                    # Auto-generated
│   ├── *_latest.pkl           # Trained model files
│   ├── *_latest_meta.json     # Model metadata
│   └── model_comparison.json  # 3-model comparison results
│
└── data/                      # Auto-generated (gitignored)
    └── workshop.db            # SQLite database
```

---

## 📊 Current Model Performance

| Model | F1 Score | AUC-ROC | Accuracy |
|-------|----------|---------|----------|
| XGBoost | 0.733 | 0.778 | 0.656 |
| Random Forest | 0.736 | 0.785 | 0.683 |
| **Logistic Regression (Winner)** | **0.748** | **0.801** | **0.683** |

> F1 is the primary metric — accuracy alone is misleading with imbalanced data.
> Winner is auto-selected by highest F1 score. Results vary by seed.
> Trained on 4 VBU schools with 16 cross-school FutureWorkshop topics.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Models** | XGBoost, Random Forest, Logistic Regression (scikit-learn) |
| **Data Balancing** | SMOTE (imbalanced-learn) |
| **Database** | SQLite via SQLAlchemy ORM |
| **Dashboard** | Streamlit + Plotly |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Serialization** | Joblib |

---

## 🔮 Future Improvements

- [ ] Integrate with college LMS / Google Forms for real data
- [ ] Student-level prediction (which specific students will attend)
- [ ] Email/notification system for low predicted turnout
- [ ] Deploy on cloud with scheduled retraining
- [ ] A/B testing for promotion strategies
- [ ] Add weather data for offline event predictions
- [ ] Per-student prediction (which specific students will attend)
- [ ] CGPA integration from university records

---

## 🗓️ Updation & Maintenance Timelines

| Phase | Frequency | Trigger | Action |
|-------|-----------|---------|--------|
| **🔄 Model Retraining** | Every semester start | New semester (Aug / Jan) | `python src/retrain.py` |
| **📊 Data Refresh** | After every 10+ events | New attendance logged | `python src/retrain.py --from-db` |
| **🔍 Performance Audit** | Monthly | Accuracy drops below threshold | Review features + threshold sweep |
| **🧹 Data Cleanup** | End of each semester | Semester ends | Archive old data, regenerate baseline |
| **🚀 Feature Updates** | As needed | New data sources (LMS, weather) | Update `feature_engineering.py`, retrain |
| **🛡️ Dependency Updates** | Quarterly | Security patches / new releases | Update `requirements.txt`, test pipeline |

**Retraining safeguard:** The retrain pipeline only deploys a new model if it beats the current one by **≥ 1 % F1 score**, preventing unnecessary swaps from random variance.

---

<div align="center">

### Built with ☕ by Rahul Purohit

*Reg: 2024SEPVUGP0079 · School of Technology — Vijaybhoomi University*

<br>

⭐ **Star this repo if you found it useful!** ⭐

</div>
