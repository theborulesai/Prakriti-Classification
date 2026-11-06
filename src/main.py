"""
main.py
========
Orchestration script for the Prakriti Classification Pipeline.
Workflow: Data Extraction -> Feature Engineering -> Model Training -> Evaluation.
"""

import os
import numpy as np
import pickle
from config import *
from feature_extraction import build_behavioral_features
from model import train_prakriti_model, save_results, save_model


def map_personality_to_prakriti(personality_labels):
    """
    Heuristic mapping from personality traits to Prakriti.
    personality_labels: (N, 6) array [ext, neu, agr, con, ope, interview]
    Returns: (N,) array of [0, 1, 2] (Vata, Pitta, Kapha)
    """
    prakriti = []
    for row in personality_labels:
        ext, neu, agr, con, ope = row[0], row[1], row[2], row[3], row[4]

        # Heuristic rules:
        # Vata:  High Openness + High Extraversion + High Neuroticism
        # Pitta: High Conscientiousness + High Extraversion + Low Neuroticism
        # Kapha: High Agreeableness + Low Neuroticism + Low Extraversion
        scores = {
            "Vata":  (ope + ext + neu) / 3,
            "Pitta": (con + ext + (1 - neu)) / 3,
            "Kapha": (agr + (1 - neu) + (1 - ext)) / 3
        }
        prakriti.append(np.argmax(list(scores.values())))

    return np.array(prakriti)


def load_or_extract_features(filenames, video_dir):
    """
    Loads cached features if available, otherwise extracts and saves them.
    This avoids re-processing 6000 videos on every run.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_path = os.path.join(FEATURES_DIR, "behavioral_features.npz")

    if os.path.exists(features_path):
        print(f"  Loading cached features from {features_path}")
        data = np.load(features_path, allow_pickle=True)
        X = data['X']
        cached_filenames = list(data['filenames'])
        # Verify cache matches current filenames
        if len(cached_filenames) == len(filenames) and cached_filenames == filenames:
            print(f"  Cached features loaded. Shape: {X.shape}")
            return X
        else:
            print("  Cache mismatch (different filenames). Re-extracting...")

    # Extract features from videos
    X = build_behavioral_features(filenames, video_dir)

    # Save to cache
    np.savez(features_path, X=X, filenames=np.array(filenames))
    print(f"  Features cached to {features_path}")

    return X


def main():
    print("=" * 60)
    print("  Prakriti Classification Pipeline")
    print("=" * 60)

    # 1. Load Annotations
    print("\n[Step 1] Loading annotations...")
    if not os.path.exists(ANNOTATION_TRAIN):
        print(f"  ERROR: Annotation file not found at {ANNOTATION_TRAIN}")
        return

    with open(ANNOTATION_TRAIN, 'rb') as f:
        ann = pickle.load(f, encoding='latin1')

    trait_list = list(ann.keys())
    filenames = sorted(list(ann[trait_list[0]].keys()))
    print(f"  Found {len(filenames)} samples in training set.")

    # 2. Extract Behavioral Features (with caching)
    print("\n[Step 2] Extracting Behavioral Features...")
    print(f"  Video directory: {TRAIN_VIDEO_DIR}")
    X_behavioral = load_or_extract_features(filenames, TRAIN_VIDEO_DIR)
    print(f"  Features shape: {X_behavioral.shape}")

    # Check how many samples are all-zeros (no video found)
    zero_count = np.sum(np.all(X_behavioral == 0, axis=1))
    if zero_count > 0:
        print(f"  [WARNING] {zero_count}/{len(filenames)} samples have all-zero features (video not found or unreadable).")

    # 3. Prepare Target Labels (Mapping personality -> Prakriti)
    print("\n[Step 3] Mapping personality traits to Prakriti labels...")
    y_personality = []
    for fn in filenames:
        row = [ann.get(trait, {}).get(fn, 0.5) for trait in
               ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness", "interview"]]
        y_personality.append(row)

    y_prakriti = map_personality_to_prakriti(np.array(y_personality))
    counts = np.bincount(y_prakriti, minlength=3)
    print(f"  Distribution: Vata={counts[0]}, Pitta={counts[1]}, Kapha={counts[2]}")

    # 4. Train and Evaluate
    print("\n[Step 4] Training model...")
    model, acc, report, cm = train_prakriti_model(X_behavioral, y_prakriti)

    # 5. Save Everything
    print("\n[Step 5] Saving results...")
    save_results(acc, report, cm)
    save_model(model)

    print("\n" + "=" * 60)
    print(f"  Pipeline Complete! Accuracy: {acc:.4f}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()