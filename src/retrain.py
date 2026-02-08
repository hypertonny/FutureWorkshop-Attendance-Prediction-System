"""
retrain.py - Model Retraining Pipeline
=======================================
This script handles retraining the model when new data is available.

How it works:
1. Load all current data from the database (or CSV)
2. Run full feature engineering pipeline
3. Train a new model
4. Compare new model's F1 score with the currently deployed model
5. Only replace the old model if the new one is better by at least 1%
   (this prevents unnecessary model swaps from random variance)
6. Log everything to the model_versions table

When to retrain:
  - After 10+ new events have been logged
  - If the model's predictions have been consistently off
  - At the start of each new semester (student behavior changes)

Usage:
  python src/retrain.py                     # retrain from CSV
  python src/retrain.py --from-db           # retrain from database
  python src/retrain.py --force             # retrain and deploy regardless
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import run_feature_pipeline
from src.train_model import prepare_data, train_xgboost, train_random_forest, save_model, get_feature_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# minimum F1 improvement to replace current model (1%)
MIN_IMPROVEMENT = 0.01


def get_current_model_metrics():
    """Load metrics of the currently deployed (best F1) model"""
    best_meta = None
    best_f1 = -1
    for model_name in ["xgboost", "random_forest"]:
        meta_path = os.path.join(MODELS_DIR, f"{model_name}_latest_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            f1 = meta.get('metrics', {}).get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_meta = meta
    return best_meta


def load_data(from_db=False):
    """Load training data from either CSV or database"""
    if from_db:
        try:
            from src.database import get_all_data_as_dataframe
            df = get_all_data_as_dataframe()
            print(f"[Retrain] Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            print(f"[Retrain] DB load failed ({e}), falling back to CSV")
    
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"[Retrain] Loaded {len(df)} rows from CSV")
    return df


def retrain(from_db=False, force=False):
    """
    Main retraining function.
    
    Steps:
    1. Load data
    2. Train new models
    3. Compare with current
    4. Deploy if better
    """
    print("\n" + "=" * 60)
    print("  RETRAINING PIPELINE")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # load current model metrics for comparison
    current = get_current_model_metrics()
    if current:
        print(f"\n[Retrain] Current model: {current['model_name']}")
        print(f"  F1: {current['metrics']['f1_score']:.4f}")
        print(f"  AUC: {current['metrics']['auc_roc']:.4f}")
        print(f"  Trained on: {current.get('timestamp', 'unknown')}")
    else:
        print("\n[Retrain] No existing model found. Will train fresh.")
        force = True  # no model exists, must deploy
    
    # load data and train
    df = load_data(from_db)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # train XGBoost (primary model)
    new_model, new_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    
    # also train RF for comparison
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    # pick the better new model
    if rf_metrics['f1_score'] > new_metrics['f1_score']:
        new_model = rf_model
        new_metrics = rf_metrics
        new_name = "random_forest"
        print("\n[Retrain] Random Forest performed better in this run")
    else:
        new_name = "xgboost"
        print("\n[Retrain] XGBoost performed better in this run")
    
    # decision: deploy or not?
    print(f"\n{'='*60}")
    print("  RETRAINING DECISION")
    print(f"{'='*60}")
    
    if force:
        print("  Mode: FORCE DEPLOY")
        should_deploy = True
    elif current is None:
        print("  No existing model - deploying new model")
        should_deploy = True
    else:
        old_f1 = current['metrics']['f1_score']
        new_f1 = new_metrics['f1_score']
        improvement = new_f1 - old_f1
        
        print(f"  Old F1: {old_f1:.4f}")
        print(f"  New F1: {new_f1:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
        print(f"  Threshold: {MIN_IMPROVEMENT}")
        
        if improvement >= MIN_IMPROVEMENT:
            should_deploy = True
            print(f"\n  >> DEPLOYING new model (improvement >= {MIN_IMPROVEMENT})")
        else:
            should_deploy = False
            print(f"\n  >> KEEPING old model (improvement < {MIN_IMPROVEMENT})")
    
    if should_deploy:
        model_path = save_model(new_model, feature_cols, new_metrics, new_name)
        
        # save feature importance
        feat_imp = get_feature_importance(new_model, feature_cols, new_name)
        if feat_imp is not None:
            feat_imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
        
        # log to database
        try:
            from src.database import log_model_version
            log_model_version(
                model_name=new_name,
                num_rows=len(X_train) + len(X_test),
                f1=new_metrics['f1_score'],
                auc=new_metrics['auc_roc'],
                acc=new_metrics['accuracy'],
                model_path=model_path
            )
        except Exception as e:
            print(f"[Warning] DB logging failed: {e}")
        
        print(f"\n  New model deployed: {new_name}")
        print(f"  F1: {new_metrics['f1_score']:.4f}")
        print(f"  AUC: {new_metrics['auc_roc']:.4f}")
    
    # log retraining attempt (even if we didn't deploy)
    log_path = os.path.join(MODELS_DIR, "retrain_log.txt")
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | "
                f"model={new_name} | "
                f"f1={new_metrics['f1_score']:.4f} | "
                f"auc={new_metrics['auc_roc']:.4f} | "
                f"deployed={should_deploy} | "
                f"rows={len(df)}\n")
    
    print(f"\n[Retrain] Log written to {log_path}")
    print("[Retrain] Done.\n")
    
    return new_model, new_metrics, should_deploy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain attendance prediction model")
    parser.add_argument("--from-db", action="store_true", help="Load data from database")
    parser.add_argument("--force", action="store_true", help="Force deploy even if not better")
    args = parser.parse_args()
    
    retrain(from_db=args.from_db, force=args.force)
