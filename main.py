"""
main.py - Entry point for the Workshop Attendance Prediction System
====================================================================
Run this script to:
  1. Auto-generate synthetic data if master_dataset.csv is missing
  2. Initialize the SQLite database
  3. Load CSV data into the database
  4. Train ML models (XGBoost, Random Forest, Logistic Regression)
  5. Save the best model

After running this, you can:
  - Launch the dashboard:  streamlit run app.py
  - Retrain the model:     python src/retrain.py
  - Make predictions:      python src/predict.py

Author: Rahul Purohit
Reg: 2024SEPVUGP0079 | School of Technology — Vijaybhoomi University
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.database import init_db, load_csv_to_db, get_all_data_as_dataframe
from src.train_model import train_all_models


def main():
    print("=" * 60)
    print("  WORKSHOP ATTENDANCE PREDICTION SYSTEM")
    print("  Setting up...")
    print("=" * 60)
    
    csv_path = os.path.join(BASE_DIR, "master_dataset.csv")
    
    # Auto-generate data if CSV is missing (fresh-from-clone support)
    if not os.path.exists(csv_path):
        print(f"\n[DataGen] master_dataset.csv not found — generating from scratch...")
        from generate_data import generate_default_dataset
        generate_default_dataset(output_path=csv_path)
        print(f"[DataGen] Dataset ready at: {csv_path}")
    
    # Step 1: Initialize database
    print("\n[Step 1/3] Initializing database...")
    init_db()
    
    # Step 2: Load CSV into database
    print("\n[Step 2/3] Loading data into database...")
    load_csv_to_db(csv_path)
    
    # verify data loaded correctly
    df = get_all_data_as_dataframe()
    print(f"  Database has {len(df)} records")
    
    # Step 3: Train models
    print("\n[Step 3/3] Training models...")
    import pandas as pd
    df_csv = pd.read_csv(csv_path)  # use CSV for training (cleaner)
    best_model, best_metrics, feature_cols = train_all_models(df_csv)
    
    # done
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print(f"\n  Best model F1 score: {best_metrics['f1_score']:.4f}")
    print(f"  Best model AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print(f"\n  To launch the dashboard, run:")
    print(f"    streamlit run app.py")
    print(f"\n  To retrain the model later:")
    print(f"    python src/retrain.py")
    print()


if __name__ == "__main__":
    main()
