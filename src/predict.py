"""
predict.py
==========
Generates Prakriti predictions on the test video set.
Uses the trained model from models/prakriti_model.pkl.
Outputs: CSV file with per-video Prakriti predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import *
from feature_extraction import build_behavioral_features


def load_or_extract_test_features(filenames, video_dir):
    """
    Loads cached test features or extracts them from videos.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_path = os.path.join(FEATURES_DIR, "test_behavioral_features.npz")

    if os.path.exists(features_path):
        print(f"  Loading cached test features from {features_path}")
        data = np.load(features_path, allow_pickle=True)
        X = data['X']
        cached_filenames = list(data['filenames'])
        if len(cached_filenames) == len(filenames) and cached_filenames == filenames:
            print(f"  Cached features loaded. Shape: {X.shape}")
            return X
        else:
            print("  Cache mismatch. Re-extracting test features...")

    X = build_behavioral_features(filenames, video_dir)
    np.savez(features_path, X=X, filenames=np.array(filenames))
    print(f"  Test features cached to {features_path}")
    return X


def main():
    print("=" * 60)
    print("  Prakriti Test Set Prediction")
    print("=" * 60)

    # 1. Load trained model
    model_path = os.path.join(MODELS_DIR, "prakriti_model.pkl")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Run main.py first.")
        return

    print(f"\n[Step 1] Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("  Model loaded successfully.")

    # 2. Discover test videos
    print(f"\n[Step 2] Scanning test video directory: {TEST_VIDEO_DIR}")
    if not os.path.exists(TEST_VIDEO_DIR):
        print(f"  ERROR: Test directory not found: {TEST_VIDEO_DIR}")
        return

    filenames = sorted([
        f for f in os.listdir(TEST_VIDEO_DIR)
        if f.lower().endswith('.mp4')
    ])
    print(f"  Found {len(filenames)} test videos.")

    if len(filenames) == 0:
        print("  No test videos found. Exiting.")
        return

    # 3. Extract / load test features
    print("\n[Step 3] Extracting Test Behavioral Features...")
    X_test = load_or_extract_test_features(filenames, TEST_VIDEO_DIR)
    print(f"  Features shape: {X_test.shape}")

    zero_count = np.sum(np.all(X_test == 0, axis=1))
    if zero_count > 0:
        print(f"  [WARNING] {zero_count}/{len(filenames)} samples have all-zero features.")

    # 4. Predict
    print("\n[Step 4] Generating Prakriti predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    prakriti_labels = [PRAKRITI_TYPES[p] for p in y_pred]
    counts = {t: prakriti_labels.count(t) for t in PRAKRITI_TYPES}
    print(f"  Prediction distribution: {counts}")

    # 5. Save predictions to CSV
    print("\n[Step 5] Saving predictions...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for i, fn in enumerate(filenames):
        rows.append({
            "filename": fn,
            "predicted_prakriti": prakriti_labels[i],
            "prakriti_code": int(y_pred[i]),
            "prob_vata": round(float(y_pred_proba[i][0]), 4),
            "prob_pitta": round(float(y_pred_proba[i][1]), 4),
            "prob_kapha": round(float(y_pred_proba[i][2]), 4),
            "hand_velocity_var": round(float(X_test[i][0]), 6),
            "head_rotation_freq": round(float(X_test[i][1]), 6),
            "path_tortuosity": round(float(X_test[i][2]), 6),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [SAVED] Predictions -> {csv_path}")

    # Print a sample of predictions
    print("\n  Sample Predictions (first 10):")
    print(df[["filename", "predicted_prakriti", "prob_vata", "prob_pitta", "prob_kapha"]].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print(f"  Test Prediction Complete! {len(filenames)} videos processed.")
    print(f"  Vata: {counts['Vata']}  |  Pitta: {counts['Pitta']}  |  Kapha: {counts['Kapha']}")
    print(f"  Output saved to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
