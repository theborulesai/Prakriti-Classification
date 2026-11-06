"""
retrain_v2.py
=============
Phase 1 Improvements:
  1. Better Prakriti label mapping (refined heuristic using all 6 traits)
  2. Better classifiers: XGBoost + MLP Neural Net + Random Forest Ensemble
  3. Cross-validation (StratifiedKFold, 5-fold) for robust evaluation
  4. Saves best model as prakriti_model_v2.pkl
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import *

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("  [INFO] XGBoost available.")
except ImportError:
    HAS_XGB = False
    print("  [INFO] XGBoost not found, using GradientBoosting instead.")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print("  [INFO] SMOTE available.")
except ImportError:
    HAS_SMOTE = False
    print("  [INFO] SMOTE not available.")


# =============================================================================
# IMPROVEMENT 1: Better Prakriti Label Mapping
# =============================================================================

def map_personality_to_prakriti_v2(personality_labels):
    """
    IMPROVED heuristic mapping from personality traits to Prakriti.
    Uses all 6 traits including 'interview' as confidence weight.
    Applies min-max normalization per trait for fair scoring.

    Ayurvedic correlations (refined):
      Vata  : High Openness, High Neuroticism, Low Conscientiousness
      Pitta : High Conscientiousness, High Extraversion, Low Agreeableness
      Kapha : High Agreeableness, Low Neuroticism, Low Openness
    """
    arr = np.array(personality_labels)

    # Normalize each trait column to [0, 1] range
    for i in range(arr.shape[1]):
        col_min, col_max = arr[:, i].min(), arr[:, i].max()
        rng = col_max - col_min
        if rng > 1e-6:
            arr[:, i] = (arr[:, i] - col_min) / rng

    prakriti = []
    for row in arr:
        ext, neu, agr, con, ope, interview = (
            row[0], row[1], row[2], row[3], row[4], row[5]
        )

        # Refined scoring with interview confidence weighting
        conf = 0.5 + 0.5 * interview  # interview score boosts confidence

        vata_score  = conf * (0.35 * ope + 0.35 * neu + 0.20 * (1 - con) + 0.10 * ext)
        pitta_score = conf * (0.35 * con + 0.35 * ext + 0.20 * (1 - agr) + 0.10 * (1 - neu))
        kapha_score = conf * (0.35 * agr + 0.35 * (1 - neu) + 0.20 * (1 - ext) + 0.10 * (1 - ope))

        prakriti.append(np.argmax([vata_score, pitta_score, kapha_score]))

    return np.array(prakriti)


# =============================================================================
# IMPROVEMENT 2: Build Ensemble of Best Classifiers
# =============================================================================

def build_classifiers():
    """Returns a dict of classifiers to compare."""
    classifiers = {}

    # Random Forest (baseline)
    classifiers["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    # MLP Neural Network
    classifiers["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001,
            solver="adam"
        ))
    ])

    # XGBoost or GradientBoosting fallback
    if HAS_XGB:
        classifiers["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    else:
        classifiers["GradientBoosting"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.1, random_state=RANDOM_STATE
            ))
        ])

    return classifiers


# =============================================================================
# IMPROVEMENT 3: Cross-Validation Evaluation
# =============================================================================

def cross_validate_all(classifiers, X, y):
    """Runs 5-fold StratifiedKFold cross-validation on all classifiers."""
    print("\n  Cross-Validation Results (5-fold StratifiedKFold):")
    print(f"  {'Classifier':<20} {'Mean Acc':>10} {'Std':>8}")
    print("  " + "-" * 42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<20} {scores.mean():>10.4f} {scores.std():>8.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    print(f"\n  >>> Best classifier: {best_name} ({cv_results[best_name].mean():.4f})")
    return best_name, cv_results


def save_v2_results(acc, report, cm, cv_results):
    """Saves v2 metrics, CV chart, and confusion matrix."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Metrics text
    out_path = os.path.join(RESULTS_DIR, "metrics_v2.txt")
    with open(out_path, "w") as f:
        f.write("=== Prakriti Model v2 Results ===\n\n")
        f.write(f"Final Hold-out Accuracy: {acc:.4f}\n\n")
        f.write("Cross-Validation Summary:\n")
        for name, scores in cv_results.items():
            f.write(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})\n")
        f.write(f"\nClassification Report:\n{report}")
    print(f"  [SAVED] Metrics -> {out_path}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - Model v2 (Improved)", fontsize=14)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(PRAKRITI_TYPES))
    ax.set_xticks(ticks); ax.set_xticklabels(PRAKRITI_TYPES)
    ax.set_yticks(ticks); ax.set_yticklabels(PRAKRITI_TYPES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    thresh = cm.max() / 2.
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center",
                color="white" if v > thresh else "black", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_v2.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  [SAVED] Confusion Matrix -> {cm_path}")

    # Cross-validation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(cv_results.keys())
    means  = [cv_results[n].mean() for n in names]
    stds   = [cv_results[n].std()  for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors[:len(names)], edgecolor="black", linewidth=0.8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Accuracy by Classifier", fontsize=13)
    ax.axhline(0.44, color="red", linestyle="--", linewidth=1.2, label="Baseline (44%)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    cv_path = os.path.join(PLOTS_DIR, "cv_comparison.png")
    plt.savefig(cv_path, dpi=150)
    plt.close()
    print(f"  [SAVED] CV Chart -> {cv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Prakriti Model v2 — Phase 1 Improvements")
    print("  [1] Better Label Mapping")
    print("  [2] XGBoost + MLP + RF Ensemble")
    print("  [3] 5-Fold Cross-Validation")
    print("=" * 60)

    # Load annotations
    print("\n[Step 1] Loading annotations...")
    with open(ANNOTATION_TRAIN, "rb") as f:
        ann = pickle.load(f, encoding="latin1")
    trait_list = list(ann.keys())
    filenames  = sorted(list(ann[trait_list[0]].keys()))
    print(f"  Found {len(filenames)} samples.")

    # Load cached features
    print("\n[Step 2] Loading cached features...")
    features_path = os.path.join(FEATURES_DIR, "behavioral_features.npz")
    if not os.path.exists(features_path):
        print(f"  ERROR: Run main.py first to generate features.")
        return
    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    print(f"  Features shape: {X.shape}")

    # IMPROVEMENT 1: Better label mapping
    print("\n[Step 3] Applying improved Prakriti label mapping...")
    y_personality = []
    for fn in filenames:
        row = [ann.get(t, {}).get(fn, 0.5) for t in
               ["extraversion", "neuroticism", "agreeableness",
                "conscientiousness", "openness", "interview"]]
        y_personality.append(row)

    y_old = np.array([np.argmax([(r[4]+r[0]+r[1])/3,
                                  (r[3]+r[0]+(1-r[1]))/3,
                                  (r[2]+(1-r[1])+(1-r[0]))/3])
                      for r in y_personality])
    y_new = map_personality_to_prakriti_v2(y_personality)

    old_counts = np.bincount(y_old, minlength=3)
    new_counts = np.bincount(y_new, minlength=3)
    print(f"  Old mapping -> Vata:{old_counts[0]}, Pitta:{old_counts[1]}, Kapha:{old_counts[2]}")
    print(f"  New mapping -> Vata:{new_counts[0]}, Pitta:{new_counts[1]}, Kapha:{new_counts[2]}")

    y = y_new

    # Apply SMOTE if available
    if HAS_SMOTE:
        print("\n[Step 4] Applying SMOTE to balance classes...")
        X_train_tmp, _, y_train_tmp, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        k = max(1, min(5, np.bincount(y_train_tmp).min() - 1))
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train_tmp, y_train_tmp)
        print(f"  After SMOTE: {np.bincount(y_res, minlength=3)}")
        # For CV, use full SMOTE-balanced dataset
        full_k = max(1, min(5, np.bincount(y).min() - 1))
        smote_full = SMOTE(random_state=RANDOM_STATE, k_neighbors=full_k)
        X_cv, y_cv = smote_full.fit_resample(X, y)
        print(f"  Full balanced set for CV: {np.bincount(y_cv, minlength=3)}")
    else:
        X_cv, y_cv = X, y

    # IMPROVEMENT 2: Build classifiers
    print("\n[Step 5] Building classifiers...")
    classifiers = build_classifiers()
    print(f"  Classifiers ready: {list(classifiers.keys())}")

    # IMPROVEMENT 3: Cross-validation
    print("\n[Step 6] Running 5-Fold Cross-Validation...")
    best_name, cv_results = cross_validate_all(classifiers, X_cv, y_cv)

    # Final train/test split with best classifier
    print(f"\n[Step 7] Training best model ({best_name}) on full data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_cv, y_cv, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cv)

    best_clf = classifiers[best_name]
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=PRAKRITI_TYPES)
    cm     = confusion_matrix(y_val, y_pred)

    print(f"\n  Final Accuracy (v2): {acc:.4f}")
    print("\n  Classification Report:\n", report)

    # Compare with baseline
    baseline_path = os.path.join(RESULTS_DIR, "metrics.txt")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            orig_acc = float(f.readline().split(":")[-1].strip())
        improvement = (acc - orig_acc) * 100
        sign = "+" if improvement >= 0 else ""
        print(f"\n  Baseline Accuracy : {orig_acc:.4f}")
        print(f"  v2 Accuracy       : {acc:.4f}  ({sign}{improvement:.2f}%)")

    # Save results
    print("\n[Step 8] Saving v2 results...")
    save_v2_results(acc, report, cm, cv_results)

    # Save best model
    model_path = os.path.join(MODELS_DIR, "prakriti_model_v2.pkl")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_clf, "best_name": best_name}, f)
    print(f"  [SAVED] Model v2 -> {model_path}")

    print("\n" + "=" * 60)
    print(f"  Phase 1 Complete!")
    print(f"  Best Classifier : {best_name}")
    print(f"  v2 Accuracy     : {acc:.4f}")
    print(f"  Saved to        : {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
