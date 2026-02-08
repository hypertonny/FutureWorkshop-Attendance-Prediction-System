<div align="center">

# 🎯 Workshop Attendance Prediction System

**Predict. Plan. Pack the room.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

*An ML-powered system that predicts student turnout for university workshops — helping organizers plan better events, allocate resources, and boost engagement.*

<br>

[🚀 Quick Start](#-quick-start) · [📊 Dashboard](#-dashboard) · [🏗️ Architecture](#️-architecture) · [🧠 How It Works](#-how-it-works) · [📁 Project Structure](#-project-structure)

</div>

---

## ❓ The Problem

University workshop organizers face a common frustration:

> *"50 students registered… but only 12 showed up."*

Without knowing expected turnout, organizers over-book venues, waste catering budgets, and can't plan logistics effectively. **This system solves that.**

---

## 💡 The Solution

A machine learning pipeline that analyzes **historical attendance patterns** across topics, speakers, timing, student behavior, and more — then predicts how many students will actually show up for a new event.

### Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Dual Model Training** | XGBoost + Random Forest — automatically picks the winner |
| 📈 **58 Engineered Features** | From 19 raw columns → rich behavioral signals |
| ♻️ **Auto-Retraining Pipeline** | Hot-swap models with 1% improvement gate |
| 📊 **Interactive Dashboard** | 5-page Streamlit app with predictions & analytics |
| 🗄️ **Scalable Database** | SQLite now, PostgreSQL-ready (just change one line) |
| ⚖️ **Imbalanced Data Handling** | SMOTE + threshold optimization for real-world skew |

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

# 4. Initialize DB + Train models
python main.py

# 5. Launch the dashboard
streamlit run app.py
```

> The dashboard opens at **http://localhost:8501** 🎉

---

## 📊 Dashboard

The Streamlit dashboard has **5 interactive pages**:

| Page | What it does |
|------|-------------|
| 🏠 **Overview** | Key metrics, attendance by topic & day-of-week charts |
| 🔮 **Predict Attendance** | Enter event details → get predicted turnout + recommendations |
| 📈 **Attendance Trends** | Monthly trends, exam impact, speaker & time slot analysis |
| 🔍 **Topic Analysis** | Deep-dive into any topic — department, semester, mode breakdown |
| ⚙️ **Model Performance** | Active model metrics, top 15 features, algorithm explanation |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  master_dataset  │────▶│  Feature Engine   │────▶│  Model Training │
│     .csv         │     │  (19 → 58 cols)  │     │  XGB + RF       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
        ┌─────────────────────────────────────────────────┘
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

Raw data has weak correlations (~0.08). The pipeline creates **4 categories** of derived features:

| Category | Examples | Why it helps |
|----------|----------|-------------|
| ⏰ **Temporal** | `semester_week`, `is_weekend`, `month` | Attendance drops late in semester |
| 👤 **Student History** | `rolling_attendance`, `streak`, `cumulative_attended` | Past behavior predicts future |
| 🔥 **Event Popularity** | `topic_popularity`, `speaker_pull`, `dept_engagement` | Some topics just hit different |
| 🔗 **Interactions** | `student_engagement_score`, `exam_is_near`, `high_promo_popular_topic` | Combined effects matter |

### Model Training

```
Raw Data → SMOTE (if imbalanced) → Train XGBoost + Random Forest
                                          ↓
                                   Threshold Sweep (0.10 → 0.60)
                                          ↓
                                   Pick best F1 → Save winner
```

- **SMOTE**: Only applied when minority class < 35%
- **Threshold Optimization**: Sweeps 0.10–0.60, picks threshold that maximizes F1
- **5-Fold Cross Validation**: Ensures scores aren't just lucky splits

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
├── main.py                    # Entry point: init DB → load data → train
├── app.py                     # Streamlit dashboard (5 pages)
├── requirements.txt           # Dependencies
├── master_dataset.csv         # Training data (5,829 rows)
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLAlchemy ORM (4 tables)
│   ├── feature_engineering.py # 19 raw → 58 features
│   ├── train_model.py         # XGBoost + RF training
│   ├── retrain.py             # Hot-retraining pipeline
│   └── predict.py             # Prediction engine
│
├── models/                    # Auto-generated (gitignored)
│   ├── *.pkl                  # Trained model files
│   └── *_meta.json            # Model metadata
│
└── data/                      # Auto-generated (gitignored)
    └── workshop.db            # SQLite database
```

---

## 📊 Current Model Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.597 |
| **AUC-ROC** | 0.716 |
| **Accuracy** | 0.635 |
| **Model** | Random Forest |

> F1 is the primary metric — accuracy alone is misleading with imbalanced data.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Models** | XGBoost, Random Forest (scikit-learn) |
| **Data Balancing** | SMOTE (imbalanced-learn) |
| **Database** | SQLite via SQLAlchemy ORM |
| **Dashboard** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |

---

## 🔮 Future Improvements

- [ ] Add real-time feedback loop from actual event outcomes
- [ ] Student-level prediction (which specific students will attend)
- [ ] Email/notification system for low predicted turnout
- [ ] Deploy on cloud (AWS/GCP) with scheduled retraining
- [ ] A/B testing for promotion strategies

---

<div align="center">

### Built with ☕ by Rahul Purohit

*CSE Department — Vijaybhoomi University*

<br>

⭐ **Star this repo if you found it useful!** ⭐

</div>
