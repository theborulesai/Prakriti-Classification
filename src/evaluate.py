"""
evaluate.py
===========
Evaluates the trained Prakriti model on the validation set.
Loads cached features if available, otherwise extracts from videos.
Outputs: validation accuracy, classification report, confusion matrix.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import *
from feature_extraction import build_behavioral_features


def map_personality_to_prakriti(personality_labels):
    """
    Heuristic mapping from personality traits to Prakriti.
    personality_labels: (N, 6) array [ext, neu, agr, con, ope, interview]
    Returns: (N,) array of [0, 1, 2] (Vata, Pitta, Kapha)
    """
    prakriti = []
    for row in personality_labels:
        ext, neu, agr, con, ope = row[0], row[1], row[2], row[3], row[4]
        scores = {
            "Vata":  (ope + ext + neu) / 3,
            "Pitta": (con + ext + (1 - neu)) / 3,
            "Kapha": (agr + (1 - neu) + (1 - ext)) / 3
        }
        prakriti.append(np.argmax(list(scores.values())))
    return np.array(prakriti)


def load_or_extract_val_features(filenames, video_dir):
    """
    Loads cached validation features or extracts them from videos.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_path = os.path.join(FEATURES_DIR, "val_behavioral_features.npz")

    if os.path.exists(features_path):
        print(f"  Loading cached validation features from {features_path}")
        data = np.load(features_path, allow_pickle=True)
        X = data['X']
        cached_filenames = list(data['filenames'])
        if len(cached_filenames) == len(filenames) and cached_filenames == filenames:
            print(f"  Cached features loaded. Shape: {X.shape}")
            return X
        else:
            print("  Cache mismatch. Re-extracting validation features...")

    X = build_behavioral_features(filenames, video_dir)
    np.savez(features_path, X=X, filenames=np.array(filenames))
    print(f"  Validation features cached to {features_path}")
    return X


def save_val_results(acc, report, cm):
    """Saves validation metrics and confusion matrix plot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Save metrics text
    out_path = os.path.join(RESULTS_DIR, "val_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"  [SAVED] Validation metrics -> {out_path}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title("Confusion Matrix - Validation Set", fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(PRAKRITI_TYPES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(PRAKRITI_TYPES)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(PRAKRITI_TYPES)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    thresh = cm.max() / 2.
    for (row, col), val in np.ndenumerate(cm):
        ax.text(col, row, format(val, 'd'),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12)

    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "val_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  [SAVED] Validation confusion matrix -> {cm_path}")


def main():
    print("=" * 60)
    print("  Prakriti Validation Evaluation")
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

    # 2. Load validation annotations
    print("\n[Step 2] Loading validation annotations...")
    if not os.path.exists(ANNOTATION_VAL):
        print(f"  ERROR: Validation annotation file not found at {ANNOTATION_VAL}")
        return

    with open(ANNOTATION_VAL, 'rb') as f:
        ann = pickle.load(f, encoding='latin1')

    trait_list = list(ann.keys())
    filenames = sorted(list(ann[trait_list[0]].keys()))
    print(f"  Found {len(filenames)} samples in validation set.")

    # 3. Extract / load validation features
    print("\n[Step 3] Extracting Validation Behavioral Features...")
    print(f"  Video directory: {VAL_VIDEO_DIR}")
    X_val = load_or_extract_val_features(filenames, VAL_VIDEO_DIR)
    print(f"  Features shape: {X_val.shape}")

    zero_count = np.sum(np.all(X_val == 0, axis=1))
    if zero_count > 0:
        print(f"  [WARNING] {zero_count}/{len(filenames)} samples have all-zero features.")

    # 4. Prepare labels
    print("\n[Step 4] Mapping personality traits to Prakriti labels...")
    y_personality = []
    for fn in filenames:
        row = [ann.get(trait, {}).get(fn, 0.5) for trait in
               ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness", "interview"]]
        y_personality.append(row)

    y_true = map_personality_to_prakriti(np.array(y_personality))
    counts = np.bincount(y_true, minlength=3)
    print(f"  Distribution: Vata={counts[0]}, Pitta={counts[1]}, Kapha={counts[2]}")

    # 5. Predict and evaluate
    print("\n[Step 5] Running predictions on validation set...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=PRAKRITI_TYPES)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Validation Accuracy: {acc:.4f}")
    print("\n  Classification Report:\n", report)

    # 6. Save results
    print("\n[Step 6] Saving validation results...")
    save_val_results(acc, report, cm)

    print("\n" + "=" * 60)
    print(f"  Validation Evaluation Complete! Accuracy: {acc:.4f}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
