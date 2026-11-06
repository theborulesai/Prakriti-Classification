"""
predict_v2.py
=============
Re-runs test predictions using the improved v2 model.
Uses cached test features (no video re-processing needed).
Outputs: results/test_predictions_v2.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *


def main():
    print("=" * 60)
    print("  Prakriti Test Predictions — Model v2")
    print("=" * 60)

    # 1. Load v2 model
    model_path = os.path.join(MODELS_DIR, "prakriti_model_v2.pkl")
    if not os.path.exists(model_path):
        print(f"  ERROR: v2 model not found at {model_path}. Run retrain_v2.py first.")
        return

    print(f"\n[Step 1] Loading v2 model from {model_path}...")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model      = saved["model"]
    best_name  = saved["best_name"]
    print(f"  Loaded: {best_name} (v2 best classifier)")

    # 2. Load cached test features
    print("\n[Step 2] Loading cached test features...")
    test_features_path = os.path.join(FEATURES_DIR, "test_behavioral_features.npz")
    if not os.path.exists(test_features_path):
        print(f"  ERROR: Cached test features not found at {test_features_path}.")
        print("  Please run predict.py first to generate and cache test features.")
        return

    data      = np.load(test_features_path, allow_pickle=True)
    X_test    = data["X"]
    filenames = list(data["filenames"])
    print(f"  Loaded {len(filenames)} test samples. Features shape: {X_test.shape}")

    # 3. Predict
    print("\n[Step 3] Generating predictions with v2 model...")
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    prakriti_labels = [PRAKRITI_TYPES[p] for p in y_pred]
    counts = {t: prakriti_labels.count(t) for t in PRAKRITI_TYPES}
    print(f"  Prediction distribution: Vata={counts['Vata']}, Pitta={counts['Pitta']}, Kapha={counts['Kapha']}")

    # 4. Load old predictions for comparison
    old_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if os.path.exists(old_csv):
        old_df = pd.read_csv(old_csv).set_index("filename")
        old_counts = old_df["predicted_prakriti"].value_counts().to_dict()
        print(f"\n  OLD model distribution: Vata={old_counts.get('Vata',0)}, "
              f"Pitta={old_counts.get('Pitta',0)}, Kapha={old_counts.get('Kapha',0)}")
        print(f"  NEW model distribution: Vata={counts['Vata']}, "
              f"Pitta={counts['Pitta']}, Kapha={counts['Kapha']}")

    # 5. Build and save output CSV
    print("\n[Step 4] Saving v2 predictions...")
    rows = []
    for i, fn in enumerate(filenames):
        row = {
            "filename":            fn,
            "predicted_prakriti":  prakriti_labels[i],
            "prakriti_code":       int(y_pred[i]),
            "prob_vata":           round(float(y_pred_proba[i][0]), 4),
            "prob_pitta":          round(float(y_pred_proba[i][1]), 4),
            "prob_kapha":          round(float(y_pred_proba[i][2]), 4),
            "hand_velocity_var":   round(float(X_test[i][0]), 6),
            "head_rotation_freq":  round(float(X_test[i][1]), 6),
            "path_tortuosity":     round(float(X_test[i][2]), 6),
            "model_version":       "v2"
        }
        rows.append(row)

    df      = pd.DataFrame(rows)
    csv_out = os.path.join(RESULTS_DIR, "test_predictions_v2.csv")
    df.to_csv(csv_out, index=False)
    print(f"  [SAVED] -> {csv_out}")

    # 6. Show sample
    print("\n  Sample Predictions (first 10):")
    print(df[["filename", "predicted_prakriti", "prob_vata", "prob_pitta", "prob_kapha"]].head(10).to_string(index=False))

    # 7. Summary comparison
    print("\n" + "=" * 60)
    print("  COMPARISON: Old Model vs v2 Model")
    print("=" * 60)
    if os.path.exists(old_csv):
        for t in PRAKRITI_TYPES:
            old_n = old_counts.get(t, 0)
            new_n = counts[t]
            diff  = new_n - old_n
            sign  = "+" if diff >= 0 else ""
            print(f"  {t:<8}: {old_n:>5} -> {new_n:>5}  ({sign}{diff})")
    print(f"\n  Total test samples: {len(filenames)}")
    print(f"  Output: {csv_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
