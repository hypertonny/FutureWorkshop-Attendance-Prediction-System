# Workshop Attendance Prediction System — Project Information

**Author:** Rahul Purohit  
**Department:** Computer Science and Engineering  
**Project Type:** Mid-term Exam / Hackathon-3  

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [Models Used and Why](#models-used-and-why)
4. [Database — What and Why](#database--what-and-why)
5. [The ML Pipeline — Step by Step](#the-ml-pipeline--step-by-step)
6. [Feature Engineering — What I Did](#feature-engineering--what-i-did)
7. [Retraining Pipeline — How It Works](#retraining-pipeline--how-it-works)
8. [Dashboard Pages](#dashboard-pages)
9. [How to Run Everything](#how-to-run-everything)
10. [Common Questions a Prof Might Ask](#common-questions-a-prof-might-ask)

---

## What This Project Does

This system predicts how many students will actually attend a planned college workshop/technical session. Colleges face a problem where 100 students register but only 20-30 show up — wasting venue, food, and organizer effort.

The system has three parts:
- **ML Model** — Predicts whether each registered student will actually attend
- **Database** — Stores student, event, and attendance data in a structured format
- **Dashboard** — Web interface where organizers enter event details and get predictions + insights

---

## Project Structure

```
hackathon-3/
├── app.py                       # Streamlit dashboard (5 pages)
├── main.py                      # Entry point: init DB → load data → train models
├── generate_data.py             # Script that adds realistic attendance patterns to data
├── master_dataset.csv           # Dataset (5,829 rows, 500 students, 100 events)
├── master_dataset_backup.csv    # Backup of original dataset
├── requirements.txt             # Python dependencies
├── information.md               # This file
│
├── data/
│   └── workshop.db              # SQLite database file
│
├── models/
│   ├── xgboost_latest.pkl       # Saved XGBoost model
│   ├── random_forest_latest.pkl # Saved Random Forest model (current winner)
│   ├── *_meta.json              # Model metadata (features used, scores, threshold)
│   ├── feature_importance.csv   # Which features matter most
│   └── retrain_log.txt          # History of all retraining attempts
│
└── src/
    ├── __init__.py
    ├── database.py              # SQLAlchemy ORM layer (SQLite, upgradable to PostgreSQL)
    ├── feature_engineering.py   # Converts 19 raw columns into 58 useful features
    ├── train_model.py           # Trains XGBoost + Random Forest, picks the best
    ├── retrain.py               # Retraining pipeline (no restart needed)
    └── predict.py               # Prediction engine for new events
```

---

## Models Used and Why

### Primary Model: XGBoost (Extreme Gradient Boosting)

**What it is:** XGBoost builds many small decision trees one after another. Each new tree focuses on fixing the mistakes the previous trees made. This is called "gradient boosting."

**Why I chose it over simpler options:**

| Option | Why Not / Why Yes |
|--------|-------------------|
| **Logistic Regression** | Too simple — assumes a straight-line boundary between "will attend" and "won't attend." Our data has complex interactions (e.g., CSE students attend AI workshops more than CIVIL students). LR misses these. |
| **KNN (K-Nearest Neighbors)** | Slow at prediction time — has to scan all training data for every prediction. Doesn't work well with mixed categorical + numerical features. Doesn't scale. |
| **Random Forest** | Good baseline. I train this too for comparison. XGBoost usually beats it because boosting learns from mistakes while RF just averages many independent trees. |
| **XGBoost** | Handles our imbalanced data natively (via `scale_pos_weight`). Built-in regularization prevents overfitting on 5,800 rows. Provides feature importance rankings. Proven best for tabular data in competitions and industry. |

**Key XGBoost Hyperparameters:**

| Parameter | Value | What it does |
|-----------|-------|--------------|
| `n_estimators` | 300 | Number of trees to build |
| `max_depth` | 5 | How deep each tree can grow (deeper = more complex but risks overfitting) |
| `learning_rate` | 0.05 | Step size for each tree's contribution (smaller = more conservative) |
| `scale_pos_weight` | ~1.83 | Tells XGBoost to pay 1.83x more attention to "attended" cases (handles class imbalance) |
| `subsample` | 0.85 | Each tree sees only 85% of data (reduces overfitting) |
| `colsample_bytree` | 0.7 | Each tree uses only 70% of features (reduces overfitting) |
| `reg_alpha` | 0.05 | L1 regularization (penalizes model complexity) |
| `reg_lambda` | 1.5 | L2 regularization (penalizes large weights) |

### Baseline Model: Random Forest

Trained alongside XGBoost for comparison. Uses `class_weight='balanced'` for imbalance handling. 300 trees, max depth 8.

### Current Performance

| Metric | XGBoost | Random Forest |
|--------|---------|---------------|
| **F1 Score** | 0.587 | **0.597** (winner) |
| **AUC-ROC** | 0.708 | **0.716** (winner) |
| **Accuracy** | 0.592 | **0.635** (winner) |

Winner is selected by **F1 Score** (not accuracy), because accuracy is misleading on imbalanced data.

### Why F1 Score, Not Accuracy?

If 65% of students don't attend, a model that always predicts "won't attend" gets 65% accuracy — but it's useless. F1 Score balances precision (of the ones we predicted will attend, how many actually did?) and recall (of all who attended, how many did we catch?). AUC-ROC measures how well the model separates the two classes overall.

### Threshold Optimization

By default, classifiers use 0.5 as the cutoff (probability > 0.5 → predict "will attend"). But with imbalanced data, lowering the threshold catches more true positives. I sweep thresholds from 0.10 to 0.60 and pick the one that maximizes F1.

Current optimal thresholds: XGBoost = 0.34, Random Forest = 0.44

---

## Database — What and Why

### What: SQLite with SQLAlchemy ORM

I use **SQLite** as the database — it's a single file (`data/workshop.db`), needs zero setup, and works everywhere. But I wrote the code using **SQLAlchemy** (an Object-Relational Mapper), which means the exact same code works with PostgreSQL, MySQL, or any other SQL database by just changing one connection string:

```python
# Current (SQLite)
engine = create_engine("sqlite:///data/workshop.db")

# Future (PostgreSQL) — just change this one line
engine = create_engine("postgresql://user:password@host:5432/workshop_db")
```

No other code changes needed. That's the scalability design.

### Why a Database Instead of Just CSV?

1. **Normalized structure** — Instead of one flat 5,829-row CSV with repeated student/event info, I split into 3 tables: `students` (500 rows), `events` (100 rows), `registrations` (5,829 rows). This avoids redundancy.
2. **Easy to update** — When a new event happens, just INSERT into `events` and `registrations` tables. With CSV, you'd append rows and risk duplicates.
3. **SQL queries** — Can run complex queries (e.g., "which student has attended the most events?") without loading everything into memory.
4. **Model tracking** — `model_versions` table tracks every trained model, its scores, and which one is currently active.

### Database Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `students` | Unique student info | student_id (PK), department, semester, club_activity_level |
| `events` | Unique event info | event_id (PK), topic, speaker_type, time_slot, mode, exam_proximity, etc. |
| `registrations` | Who registered + attended | student_id, event_id, attended (0/1), registration_timing |
| `model_versions` | Model deployment history | model_name, f1_score, auc_roc, model_path, is_active flag |

---

## The ML Pipeline — Step by Step

Here's what happens from raw data to prediction, in order:

### Step 1: Data Loading
- Read `master_dataset.csv` (5,829 rows × 19 columns)
- 500 unique students, 100 unique events
- Each row = one student registered for one event
- Target column: `attended` (0 = didn't show up, 1 = showed up)
- Current attendance rate: ~35%

### Step 2: Feature Engineering (19 → 58 features)
- Take the 19 raw columns and derive 39 additional features
- Four categories: temporal, student history, event popularity, interactions
- (Detailed in the Feature Engineering section below)

### Step 3: Data Splitting
- 80% training, 20% testing (stratified — same attendance ratio in both)
- Training set: 4,663 rows, Test set: 1,166 rows
- Test set is NEVER touched during training — only used for final evaluation

### Step 4: SMOTE (if needed)
- If the minority class (attended=1) is less than 35% of training data, SMOTE generates synthetic positive examples
- SMOTE = Synthetic Minority Oversampling Technique — creates new "attended" rows by interpolating between existing ones
- Only applied to training data. Test data stays untouched.

### Step 5: Model Training
- Train XGBoost with gradient boosting and regularization
- Train Random Forest with balanced class weights
- Both models learn to map 58 features → probability of attending

### Step 6: Threshold Optimization
- Sweep thresholds 0.10 → 0.60 on each model
- Pick the threshold that maximizes F1 Score
- This gives better results than the default 0.5 cutoff

### Step 7: Evaluation
- Calculate Accuracy, F1 Score, AUC-ROC on the test set
- Print confusion matrix and classification report
- 5-fold cross-validation on XGBoost for stability check

### Step 8: Model Selection
- Compare XGBoost and Random Forest by F1 Score
- The winner becomes the active model

### Step 9: Save
- Both models saved as `.pkl` files (pickle/joblib serialization)
- Metadata saved as `.json` (features used, scores, threshold, timestamp)
- Feature importance exported to CSV for the dashboard
- Winner logged to `model_versions` database table

### Pipeline Diagram

```
CSV Data (5,829 rows)
    │
    ▼
Feature Engineering (19 → 58 features)
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ├── Training Data ──→ SMOTE (if imbalanced) ──→ Train XGBoost
    │                                              ──→ Train Random Forest
    │
    ├── Test Data ──→ Evaluate both models
    │
    ▼
Compare F1 Scores → Pick Winner → Save to disk + log to DB
```

---

## Feature Engineering — What I Did

The raw dataset had 19 columns but very weak correlations with attendance (max ~0.13). I engineered 39 additional features across 4 categories to give the model stronger signals.

### Category 1: Temporal Features (from event_date)

| Feature | How | Why |
|---------|-----|-----|
| `month` | Extract month from date | Seasonal patterns (August start vs December exam season) |
| `is_weekend` | 1 if Saturday/Sunday | Weekend events have lower turnout |
| `semester_week` | Which week of the semester | Engagement drops as semester progresses |

### Category 2: Student History Features (per-student tracking)

| Feature | How | Why |
|---------|-----|-----|
| `student_rolling_attendance` | Expanding mean of past attended values per student, shifted by 1 to prevent data leakage | A student with 80% past attendance will likely attend next |
| `student_cumulative_attended` | Cumulative count of past events attended | Measures overall engagement level |
| `student_total_events` | How many events the student has been invited to | Context for the rate |
| `attendance_streak` | Consecutive attend (+) or miss (-) count | Momentum: attending begets attending |

**Data leakage prevention:** All history features use only past data (shifted by 1 event). The model never sees the current event's attendance when building features.

### Category 3: Event Popularity Features (historical rates)

| Feature | How | Why |
|---------|-----|-----|
| `topic_popularity` | Historical attendance rate per topic | Data Science (27%) vs IoT (14%) — different draw |
| `speaker_pull` | Historical attendance rate per speaker type | Industry (44%) vs Faculty (28%) |
| `dept_engagement` | Historical attendance rate per department | Some departments are more active |
| `timeslot_popularity` | Historical attendance rate per time slot | Afternoon (40%) vs Morning (30%) |
| `day_popularity` | Historical attendance rate per day | Weekday vs weekend patterns |
| `registration_density` | num_registrations / average | Hype factor — popular events have momentum |

### Category 4: Interaction Features (combined signals)

| Feature | How | Why |
|---------|-----|-----|
| `exam_is_near` | Binary: exam_proximity == 1 | Cleaner signal than ordinal 1/2/3 |
| `is_long_workshop` | Binary: duration > 120 min | Long workshops lose people |
| `high_promo_popular_topic` | (promotion == High) × topic_popularity | Highly promoted popular topics should do best |
| `club_activity_numeric` | High=3, Medium=2, Low=1 | Ordinal encoding |
| `student_engagement_score` | rolling_attendance × club_activity_numeric | Combined student engagement metric |

### Encoding

- **Ordinal encoding:** registration_timing (Early=3, Medium=2, Late=1), promotion_level, mode
- **One-hot encoding:** department, topic, speaker_type, time_slot, day_of_week (with drop_first to avoid multicollinearity)

---

## Retraining Pipeline — How It Works

### Do I Need to Shut Down the Project to Retrain?

**No.** The retraining pipeline is designed to work while the dashboard is running. Here's why:

1. The dashboard loads the model from `models/random_forest_latest.pkl` at startup (cached)
2. The retrain script trains a new model and saves it as a NEW timestamped file first
3. Only after confirming the new model is better, it overwrites the `_latest.pkl` file
4. The dashboard picks up the new model on the next page load/prediction call

So the workflow is:
```
Terminal 1: streamlit run app.py          ← dashboard stays running
Terminal 2: python src/retrain.py         ← retrain in parallel, no conflict
```

### Retraining Decision Logic

```
1. Load current deployed model's F1 score from metadata
2. Train new XGBoost + Random Forest on latest data
3. Pick the better new model (by F1)
4. Compare: new_f1 - old_f1 >= 0.01 (1% improvement threshold)?
   ├── YES → Save new model, overwrite _latest.pkl, log to DB
   └── NO  → Keep old model, log attempt to retrain_log.txt
```

The 1% threshold prevents unnecessary model swaps from random variance (same data might produce slightly different scores due to randomness in trees).

### CLI Commands

```bash
python src/retrain.py              # Retrain from CSV, deploy only if better
python src/retrain.py --from-db    # Retrain from database data
python src/retrain.py --force      # Force deploy even if not better
```

### When Should I Retrain?

- After 10+ new events have been logged with actual attendance
- If the model's predictions are consistently off
- At the start of a new semester (student behavior changes)
- After significant changes in workshop formats

### Retraining Log

Every retrain attempt (successful or not) is logged to `models/retrain_log.txt`:
```
2026-02-08T17:44:53 | model=random_forest | f1=0.5974 | auc=0.7156 | deployed=True | rows=5829
```

---

## Dashboard Pages

The Streamlit dashboard has 5 pages:

| Page | What It Shows |
|------|---------------|
| **Overview** | 4 key metrics (total events, students, registrations, avg attendance rate). Charts: attendance by topic, attendance by day-of-week. Active model card with F1 and AUC scores. |
| **Predict Attendance** | Form to input planned event details (topic, speaker, time, mode, etc.). Clicking "Predict" runs the ML model and shows: predicted attendees, predicted rate, confidence level, venue recommendation, historical topic reference. |
| **Attendance Trends** | Monthly registration vs actual attendance over time. Monthly attendance rate. Impact of exam proximity. Impact of speaker type. Day × time slot heatmap. |
| **Topic Analysis** | Filter by topic. Per-topic stats: events, registrations, attendance rate. Department breakdown. Semester breakdown. Online vs offline comparison. Club activity impact. |
| **Model Performance** | Active model metrics (Accuracy, F1, AUC). Top 15 feature importance chart. "How the Model Works" explanation. Last 10 retraining log entries. |

---

## How to Run Everything

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate realistic data (if using original synthetic data)
python generate_data.py

# 3. Initialize database + train models
python main.py

# 4. Launch dashboard
streamlit run app.py
```

### Day-to-Day Usage

```bash
# Launch dashboard
streamlit run app.py

# Retrain model (in a separate terminal, dashboard stays up)
python src/retrain.py

# Force retrain and deploy
python src/retrain.py --force
```

---

## Common Questions a Prof Might Ask

### Q: Why XGBoost and not a neural network?

Our dataset has 5,829 rows with tabular (structured) data. Neural networks need much larger datasets (10K+) and are designed for unstructured data like images or text. XGBoost is the proven best algorithm for tabular data — it won most Kaggle competitions involving structured data. For our data size and type, XGBoost is the right choice.

### Q: How do you handle class imbalance?

Three ways:
1. **scale_pos_weight** in XGBoost — tells the model to pay more attention to the minority class (attended=1)
2. **SMOTE** — generates synthetic positive samples when minority class < 35%
3. **Threshold tuning** — instead of the default 0.5 cutoff, I find the optimal probability threshold by sweeping 0.10–0.60 and picking the one that maximizes F1

### Q: How do you prevent data leakage?

- All student history features (rolling attendance, streaks) are computed using only past events, shifted by 1 position
- SMOTE is applied only to training data, never to the test set
- Test set is stratified and held out before any processing

### Q: What is data leakage?

Data leakage happens when the model accidentally sees information from the future during training. Example: if I used a student's overall attendance rate (including future events) to predict the current event, the model would cheat — it would know the student attended a future event. I prevent this by only using past data.

### Q: Why not just accuracy?

With 65% non-attendance, a model that always predicts "won't attend" gets 65% accuracy but is completely useless — it never identifies who WILL attend. F1 Score captures both precision and recall — it tells us how good the model is at correctly identifying actual attendees.

### Q: Why SQLite and not PostgreSQL?

For a prototype/mid-term project, SQLite is perfect — it's a single file, no server setup, works on any machine. But I used SQLAlchemy (an ORM), so switching to PostgreSQL in production only requires changing one connection string. Zero code changes. This is the scalability strategy.

### Q: How does the prediction work for a new event?

1. The organizer enters event details (topic, speaker, timing, etc.) into the dashboard
2. The system samples N students from the student pool (N = expected registrations)
3. It creates virtual "registration" rows for these students with the new event details
4. It runs full feature engineering (computing each student's historical attendance, topic popularity, etc.)
5. The trained model predicts each student's probability of attending
6. It sums up the predictions to give a total expected attendance count

### Q: What is feature engineering and why did you need it?

The raw data had 19 columns but very weak correlation with attendance (max 0.13). Feature engineering means creating new columns from existing ones that capture actual behavioral patterns. For example:
- A student who attended 8 out of 10 past events has a `student_rolling_attendance` of 0.8 — this is much more predictive than the raw `semester` or `department` column
- A topic like "Data Science" with historical 27% attendance has a `topic_popularity` of 0.27 — this encodes real-world popularity

I went from 19 raw features to 58 engineered features. This is why the model works.

### Q: Can you retrain without stopping the dashboard?

Yes. The retrain script saves the new model as a new timestamped file, compares it with the current model, and only replaces the active model file if the new one is better. The dashboard loads the model fresh on each prediction call, so it automatically picks up the new model without restarting.

### Q: What's the difference between training and retraining?

- **Training** (initial): Run `python main.py` — sets up everything from scratch, creates database, trains first model
- **Retraining**: Run `python src/retrain.py` — loads all current data (including any new events), trains a new model, compares it with the deployed model, replaces only if better. No downtime.

### Q: What happens if new model is worse after retraining?

The retrain pipeline has a safety check — it only deploys the new model if F1 improves by at least 1% over the current model. If not, it keeps the old model and logs the failed attempt to `retrain_log.txt`. This prevents model degradation from random variance.

### Q: How would you improve this further?

- Use real data instead of synthetic (integrate with Google Forms or college LMS)
- Add student-level features like CGPA, friend network (peer influence)
- Track weather data for offline events
- Use LightGBM as an additional model (faster, native categorical support)
- Add email/SMS notification integration for low-prediction events
- Deploy as a proper web app with FastAPI backend + Streamlit frontend
- Switch to PostgreSQL for multi-user concurrent access

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.2 | Data manipulation and analysis |
| numpy | 1.26.4 | Numerical operations |
| scikit-learn | 1.5.1 | ML utilities, Random Forest, metrics, train-test split |
| xgboost | 2.1.1 | Primary gradient boosting classifier |
| imbalanced-learn | 0.12.3 | SMOTE for handling class imbalance |
| matplotlib | 3.9.2 | Plotting and visualizations |
| seaborn | 0.13.2 | Heatmaps and styled statistical plots |
| streamlit | 1.38.0 | Interactive web dashboard |
| sqlalchemy | 2.0.35 | ORM for database operations |
| joblib | 1.4.2 | Model serialization (save/load .pkl files) |
