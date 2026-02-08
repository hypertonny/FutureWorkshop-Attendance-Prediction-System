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
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import run_feature_pipeline
from src.train_model import (prepare_data, train_xgboost, train_random_forest, 
                             train_logistic_regression, save_model, get_feature_importance)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# minimum F1 improvement to replace current model (1%)
MIN_IMPROVEMENT = 0.01


def get_current_model_metrics():
    """Load metrics of the currently deployed (best F1) model"""
    # check comparison file first for the winner
    comp_path = os.path.join(MODELS_DIR, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp = json.load(f)
        winner = comp.get('winner', 'xgboost')
        meta_path = os.path.join(MODELS_DIR, f"{winner}_latest_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
    
    # fallback: check each model
    for name in ['xgboost', 'random_forest', 'logistic_regression']:
        meta_path = os.path.join(MODELS_DIR, f"{name}_latest_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
    return None


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
    
    # load data and train all 3 models
    df = load_data(from_db)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # train all 3 models
    results = {}
    
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['logistic_regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # pick the best new model by F1
    new_name = max(results, key=lambda k: results[k]['metrics']['f1_score'])
    new_model = results[new_name]['model']
    new_metrics = results[new_name]['metrics']
    
    print(f"\n[Retrain] Best new model: {new_name} (F1={new_metrics['f1_score']:.4f})")
    
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
        # save all 3 models
        for name, res in results.items():
            save_model(res['model'], feature_cols, res['metrics'], name)
        
        # save feature importance from winner
        feat_imp = get_feature_importance(new_model, feature_cols, new_name)
        if feat_imp is not None:
            feat_imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
        
        # save comparison results
        comparison = {}
        for name, res in results.items():
            comparison[name] = res['metrics']
        comparison['winner'] = new_name
        comp_path = os.path.join(MODELS_DIR, "model_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # log to database
        try:
            from src.database import log_model_version
            log_model_version(
                model_name=new_name,
                num_rows=len(X_train) + len(X_test),
                f1=new_metrics['f1_score'],
                auc=new_metrics['auc_roc'],
                acc=new_metrics['accuracy'],
                model_path=os.path.join(MODELS_DIR, f"{new_name}_latest.pkl")
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
