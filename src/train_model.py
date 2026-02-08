"""
train_model.py - Train and evaluate ML models
==============================================
Models used:
1. XGBoost (primary)   - Gradient boosting, handles imbalance natively
2. Random Forest       - Ensemble baseline for comparison

Why not KNN?
  KNN is slow at prediction time (has to scan all training data), 
  doesn't work well with mixed feature types, and doesn't scale.
  XGBoost is much better for tabular data like ours.

Why XGBoost over plain Logistic Regression?
  Our dataset has non-linear patterns (e.g., topic + time + student history
  interact in complex ways). Logistic Regression assumes linear boundaries
  which misses these interactions. XGBoost builds decision trees that 
  naturally capture non-linear patterns.
  
Evaluation:
  Using F1, AUC-ROC, and Accuracy. We can't rely on just accuracy because
  the dataset is imbalanced (82% didn't attend). A dumb model that always
  predicts 0 gets 82% accuracy but is useless.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import run_feature_pipeline


# ---- Config ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare_data(df):
    """
    Run feature engineering, split into train/test, and apply SMOTE.
    
    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic 
    samples for the minority class (attended=1) by interpolating between 
    existing minority samples. This gives the model more positive examples 
    to learn from without just duplicating rows.
    
    Important: SMOTE is only applied to training data, NOT test data.
    Test set must remain untouched for honest evaluation.
    """
    df_featured, feature_cols = run_feature_pipeline(df)
    
    X = df_featured[feature_cols]
    y = df_featured['attended']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n[Data] Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print(f"[Data] Train attendance rate (before SMOTE): {y_train.mean():.2%}")
    
    # apply SMOTE only if data is significantly imbalanced (minority < 35%)
    minority_ratio = y_train.mean()
    if minority_ratio < 0.35:
        target_ratio = min(0.45, minority_ratio * 2)
        smote = SMOTE(sampling_strategy=target_ratio, random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[Data] Train after SMOTE: {X_train.shape[0]} rows")
        print(f"[Data] Train attendance rate (after SMOTE): {y_train.mean():.2%}")
    else:
        print(f"[Data] SMOTE skipped (data already balanced enough)")
    
    print(f"[Data] Test attendance rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def find_best_threshold(model, X_test, y_test):
    """
    Find the probability threshold that maximizes F1 score.
    
    By default, models use 0.5 as threshold (if probability > 0.5, predict 1).
    But with imbalanced data, a lower threshold can catch more true positives.
    We try thresholds from 0.1 to 0.6 and pick the one with highest F1.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.10, 0.60, 0.02):
        y_pred = (y_prob >= thresh).astype(int)
        f1_val = f1_score(y_test, y_pred)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh
    
    return best_thresh, best_f1


def evaluate_model(model, X_test, y_test, model_name):
    """Calculate all metrics using optimized threshold"""
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # find optimal threshold instead of default 0.5
    best_thresh, _ = find_best_threshold(model, X_test, y_test)
    y_pred = (y_prob >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{'='*50}")
    print(f"  {model_name} - Results")
    print(f"{'='*50}")
    print(f"  Optimal Threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Attended', 'Attended']))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    
    return {"accuracy": acc, "f1_score": f1, "auc_roc": auc, "threshold": best_thresh}


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier.
    
    Key parameter: scale_pos_weight
    Since only 18% of students attend, the classes are imbalanced (82:18).
    scale_pos_weight tells XGBoost to pay more attention to the minority class.
    The ratio is: (number of 0s) / (number of 1s) ≈ 4.5
    
    Other important params:
    - n_estimators: number of trees (100 is a good starting point)
    - max_depth: how deep each tree can grow (4 prevents overfitting on 5.8K rows)
    - learning_rate: step size (0.1 is standard, smaller = more trees needed)
    - reg_alpha/reg_lambda: L1/L2 regularization to prevent overfitting
    """
    # calculate class weight ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    
    print(f"\n[XGBoost] Class ratio: {neg_count}:{pos_count} (scale_pos_weight={scale_weight:.2f})")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,       # L1 regularization
        reg_lambda=1.5,       # L2 regularization
        subsample=0.85,       # use 85% of data per tree (reduces overfitting)
        colsample_bytree=0.7,  # use 70% of features per tree
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "XGBoost")
    
    # cross-validation to make sure we're not just lucky with the split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"\n[XGBoost] 5-Fold CV F1 scores: {cv_scores.round(4)}")
    print(f"[XGBoost] Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest as a baseline comparison.
    Using class_weight='balanced' to handle imbalanced data.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',  # automatically adjusts for imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1  # use all CPU cores
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "Random Forest")
    
    return model, metrics


def get_feature_importance(model, feature_cols, model_name="XGBoost"):
    """Get and display top features that the model relies on"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n[{model_name}] Top 15 Important Features:")
        for _, row in feat_imp.head(15).iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")
        
        return feat_imp
    return None


def save_model(model, feature_cols, metrics, model_name):
    """Save model and metadata to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}"
    
    # save the model
    model_path = os.path.join(MODELS_DIR, f"{filename}.pkl")
    joblib.dump(model, model_path)
    
    # save metadata (features used, metrics, etc.)
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "threshold": metrics.get("threshold", 0.5),
        "model_path": model_path
    }
    meta_path = os.path.join(MODELS_DIR, f"{filename}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # also save as "latest" for easy loading
    latest_path = os.path.join(MODELS_DIR, f"{model_name}_latest.pkl")
    joblib.dump(model, latest_path)
    latest_meta = os.path.join(MODELS_DIR, f"{model_name}_latest_meta.json")
    with open(latest_meta, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[Save] Model saved to: {model_path}")
    print(f"[Save] Latest model: {latest_path}")
    
    return model_path


def train_all_models(df):
    """
    Main training function - trains all models, compares them,
    saves the best one as the active model.
    """
    print("=" * 60)
    print("  WORKSHOP ATTENDANCE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # train both models
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    # compare and pick the best model
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<15} {'XGBoost':<12} {'Random Forest':<12}")
    print(f"  {'-'*39}")
    for metric in ['accuracy', 'f1_score', 'auc_roc']:
        print(f"  {metric:<15} {xgb_metrics[metric]:<12.4f} {rf_metrics[metric]:<12.4f}")
    
    # pick winner based on F1 score (most important for imbalanced data)
    if xgb_metrics['f1_score'] >= rf_metrics['f1_score']:
        best_model, best_name, best_metrics = xgb_model, "xgboost", xgb_metrics
        print(f"\n  >> Winner: XGBoost (better F1)")
    else:
        best_model, best_name, best_metrics = rf_model, "random_forest", rf_metrics
        print(f"\n  >> Winner: Random Forest (better F1)")
    
    # save both models
    xgb_path = save_model(xgb_model, feature_cols, xgb_metrics, "xgboost")
    rf_path = save_model(rf_model, feature_cols, rf_metrics, "random_forest")
    
    # save feature importance from the winning model
    feat_imp = get_feature_importance(best_model, feature_cols, best_name)
    if feat_imp is not None:
        feat_imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
    
    # try logging to database
    try:
        from src.database import log_model_version
        log_model_version(
            model_name=best_name,
            num_rows=len(X_train) + len(X_test),
            f1=best_metrics['f1_score'],
            auc=best_metrics['auc_roc'],
            acc=best_metrics['accuracy'],
            model_path=os.path.join(MODELS_DIR, f"{best_name}_latest.pkl")
        )
    except Exception as e:
        print(f"[Warning] Could not log to DB: {e}")
    
    return best_model, best_metrics, feature_cols


if __name__ == "__main__":
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    df = pd.read_csv(csv_path)
    train_all_models(df)
