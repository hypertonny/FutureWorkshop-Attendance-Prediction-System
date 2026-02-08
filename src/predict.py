"""
predict.py - Make predictions using the trained model
=====================================================
This module loads the latest trained model and makes predictions
for new/upcoming events.

Used by:
  - The Streamlit dashboard (app.py) for the prediction interface
  - Can also be used standalone for quick predictions
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import run_feature_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_latest_model(model_name=None):
    """
    Load the latest saved model and its metadata.
    If model_name is not specified, picks the one with the best F1 score.
    """
    if model_name is None:
        # check comparison file first
        comp_path = os.path.join(MODELS_DIR, "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, 'r') as f:
                comp = json.load(f)
            model_name = comp.get('winner', 'xgboost')
        else:
            # fallback: pick the best model by checking metadata files
            best_f1 = -1
            best_name = "xgboost"  # default fallback
            for name in ["xgboost", "random_forest", "logistic_regression"]:
                meta_path = os.path.join(MODELS_DIR, f"{name}_latest_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    f1 = meta.get('metrics', {}).get('f1_score', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_name = name
            model_name = best_name
    model_path = os.path.join(MODELS_DIR, f"{model_name}_latest.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{model_name}_latest_meta.json")
    
    if not os.path.exists(model_path):
        # try other models as fallback
        for alt in ["xgboost", "random_forest", "logistic_regression"]:
            alt_path = os.path.join(MODELS_DIR, f"{alt}_latest.pkl")
            alt_meta = os.path.join(MODELS_DIR, f"{alt}_latest_meta.json")
            if os.path.exists(alt_path):
                model_path = alt_path
                meta_path = alt_meta
                break
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run train_model.py first.")
    
    model = joblib.load(model_path)
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"[Predict] Loaded model from: {model_path}")
    return model, metadata


def predict_single_event(event_details, historical_df):
    """
    Predict attendance for a single upcoming event.
    
    Parameters:
    -----------
    event_details : dict
        Details of the planned event. Example:
        {
            'topic': 'Data Science',
            'speaker_type': 'Industry',
            'day_of_week': 'Wednesday',
            'time_slot': 'Afternoon (2-4 PM)',
            'duration_minutes': 90,
            'mode': 'Offline',
            'exam_proximity': 3,
            'promotion_level': 'High',
            'num_registrations': 60,
            'event_date': '2026-03-15'
        }
    historical_df : pd.DataFrame
        The full historical dataset (needed for feature engineering context)
    
    Returns:
    --------
    dict with predicted attendance count, probability, and confidence
    """
    model, metadata = load_latest_model()
    feature_cols = metadata.get('feature_columns', [])
    
    # we need to create rows for "synthetic" students to predict per-student attendance
    # then aggregate to get expected total attendance
    
    # get unique students from historical data
    student_cols = ['student_id', 'department', 'semester', 'club_activity_level']
    if 'cgpa' in historical_df.columns:
        student_cols.append('cgpa')
    students = historical_df[student_cols].drop_duplicates('student_id')
    
    num_registered = event_details.get('num_registrations', 50)
    
    # sample students (simulating who would register)
    if len(students) > num_registered:
        sampled_students = students.sample(n=num_registered, random_state=42)
    else:
        sampled_students = students
    
    # create prediction rows
    pred_rows = []
    for _, student in sampled_students.iterrows():
        row = {
            'student_id': student['student_id'],
            'event_id': 'E_NEW',
            'department': student['department'],
            'semester': student['semester'],
            'club_activity_level': student['club_activity_level'],
            'registration_timing': 'On-time',  # most common timing (~45%)
            'attended': 0,  # placeholder
            'past_attendance_rate': 0.0,
            'past_events_count': 0,
            **event_details
        }
        if 'cgpa' in student.index:
            row['cgpa'] = student['cgpa']
        pred_rows.append(row)
    
    pred_df = pd.DataFrame(pred_rows)
    
    # combine with historical data for feature engineering context
    full_df = pd.concat([historical_df, pred_df], ignore_index=True)
    
    # run feature engineering
    featured_df, all_features = run_feature_pipeline(full_df)
    
    # extract only the new event rows
    new_event_mask = featured_df['event_id'] == 'E_NEW'
    X_new = featured_df.loc[new_event_mask].copy()
    
    # handle any missing columns (features present in training but not here)
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_cols]
    
    # predict using optimized threshold from training
    threshold = metadata.get('threshold', metadata.get('metrics', {}).get('threshold', 0.5))
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # aggregate results
    predicted_attendees = int(predictions.sum())
    avg_probability = float(probabilities.mean())
    
    result = {
        'predicted_attendees': predicted_attendees,
        'total_registered': num_registered,
        'attendance_rate': predicted_attendees / max(num_registered, 1),
        'avg_probability': avg_probability,
        'confidence': 'High' if avg_probability > 0.3 else ('Medium' if avg_probability > 0.15 else 'Low')
    }
    
    return result


def predict_batch(df):
    """
    Predict attendance for all rows in a dataframe.
    Used for evaluating the model on historical data.
    """
    model, metadata = load_latest_model()
    feature_cols = metadata.get('feature_columns', [])
    
    featured_df, _ = run_feature_pipeline(df)
    X = featured_df.copy()
    
    # handle missing columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities


if __name__ == "__main__":
    # quick test
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    df = pd.read_csv(csv_path)
    
    test_event = {
        'topic': 'Data Science',
        'speaker_type': 'Industry',
        'day_of_week': 'Wednesday',
        'time_slot': 'Afternoon (2-4 PM)',
        'duration_minutes': 90,
        'mode': 'Offline',
        'exam_proximity': 3,
        'promotion_level': 'High',
        'num_registrations': 60,
        'event_date': '2026-03-15'
    }
    
    result = predict_single_event(test_event, df)
    print("\n[Predict] Test Prediction:")
    print(f"  Registered: {result['total_registered']}")
    print(f"  Predicted attendees: {result['predicted_attendees']}")
    print(f"  Predicted rate: {result['attendance_rate']:.1%}")
    print(f"  Confidence: {result['confidence']}")
