# Prakriti Classification Using Behavioral Feature Extraction from Videos

> IIIT Dharwad | Mini Project-II | 7th Semester | Aug-Nov 2026

## Team
| Name | USN |
|------|-----|
| Sainath Borule | 23BCS115 |
| Pankaj Kumar | 23BCS091 |
| Shaik Mohammed Samiulla | 23BCS120 |
| Kislay Singh | 23BCS066 |

**Guide:** Dr. Chinmayananda A, Assistant Professor, CSSE, IIIT Dharwad

---

## About

This project classifies a person's **Prakriti** (Ayurvedic body-mind type: Vata, Pitta, or Kapha) automatically from interview videos using Computer Vision and Machine Learning.

- Uses **Google MediaPipe** to extract hand, face and pose landmarks
- Extracts 3 behavioral features: Hand Velocity Variance, Head Rotation Frequency, Path Tortuosity
- Maps Big Five personality scores to Prakriti using Ayurveda-grounded heuristics
- Applies **SMOTE** for class balancing and **5-fold cross-validation** for model selection
- Best model: **Random Forest v2 â€” 47.80% accuracy** (+5.13% over baseline)

---

## Results

| Model | Accuracy |
|-------|----------|
| Baseline (v1) | 42.67% |
| Improved (SMOTE) | 43.00% |
| **v2 (Final)** | **47.80%** |
| Validation Set | 43.95% |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py          # Train baseline
python src/evaluate.py      # Validate
python src/retrain_v2.py    # Improved training
python src/predict_v2.py    # Test predictions
```

## Tech Stack
Python 3.12 | MediaPipe | scikit-learn | XGBoost | OpenCV | imbalanced-learn