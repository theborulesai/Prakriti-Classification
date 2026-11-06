"""
model.py
========
Implements a classifier to map behavioral features to Prakriti types.
Uses RandomForest with optional SMOTE oversampling for class imbalance.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import *

# Optional: import SMOTE if available for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def train_prakriti_model(X, y):
    """
    Trains a Random Forest Classifier to predict Prakriti.
    X: (N, 3) array of [hand_var, head_freq, path_tort]
    y: (N,) array of Prakriti classes (0: Vata, 1: Pitta, 2: Kapha)
    """
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Address Class Imbalance using SMOTE if available
    if HAS_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  [SMOTE] Balanced training set: {np.bincount(y_train_res)}")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("  [INFO] imblearn not installed. Using balanced class weights instead.")

    # Pipeline: Scale -> RF Classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ))
    ])

    pipeline.fit(X_train_res, y_train_res)

    # Evaluation
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=PRAKRITI_TYPES)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n  [MODEL] Validation Accuracy: {acc:.4f}")
    print("\n  Classification Report:\n", report)

    return pipeline, acc, report, cm


def save_results(acc, report, cm):
    """Saves metrics and confusion matrix plot to the results folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Save accuracy and report
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Final Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - Prakriti Classification", fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(PRAKRITI_TYPES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(PRAKRITI_TYPES)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(PRAKRITI_TYPES)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Annotate cells with counts
    thresh = cm.max() / 2.
    for (row, col), val in np.ndenumerate(cm):
        ax.text(col, row, format(val, 'd'),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  [SAVED] Results saved to {RESULTS_DIR}")


def save_model(model, path=None):
    """Saves the trained model as a pickle file."""
    if path is None:
        path = os.path.join(MODELS_DIR, "prakriti_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [SAVED] Model -> {path}")