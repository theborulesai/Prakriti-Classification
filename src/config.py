"""
config.py - Central configuration for the Prakriti Classification project
"""
import os

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT   = r"D:\PROJECTS\Prakriti_Classification_Final"
# Dataset root: videos are directly under data/raw/ (train/, validation/, test/)
DATASET_ROOT   = os.path.join(PROJECT_ROOT, "data", "raw")

TRAIN_VIDEO_DIR = os.path.join(DATASET_ROOT, "train")
VAL_VIDEO_DIR   = os.path.join(DATASET_ROOT, "validation")
TEST_VIDEO_DIR  = os.path.join(DATASET_ROOT, "test")

TRAIN_META_CSV  = os.path.join(TRAIN_VIDEO_DIR, "metadata.csv")
VAL_META_CSV    = os.path.join(VAL_VIDEO_DIR,   "metadata.csv")
TEST_META_CSV   = os.path.join(TEST_VIDEO_DIR,  "metadata.csv")

ANNOTATION_TRAIN = os.path.join(PROJECT_ROOT, "data", "annotations", "annotation_training.pkl")
ANNOTATION_VAL   = os.path.join(PROJECT_ROOT, "data", "annotations", "annotation_validation.pkl")

FEATURES_DIR   = os.path.join(PROJECT_ROOT, "data", "features")
MP_MODELS_DIR  = os.path.join(PROJECT_ROOT, "data", "mp_models")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR      = os.path.join(RESULTS_DIR,  "plots")

# -- MediaPipe model paths ----------------------------------------------------
HAND_MODEL_PATH = os.path.join(MP_MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MP_MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MP_MODELS_DIR, "pose_landmarker_lite.task")

# -- Prakriti mapping ---------------------------------------------------------
# Vata  : High Openness + High Extraversion + High Neuroticism
# Pitta : High Conscientiousness + High Extraversion + Low Neuroticism
# Kapha : High Agreeableness + Low Neuroticism + Low Extraversion
PRAKRITI_TYPES = ["Vata", "Pitta", "Kapha"]

# -- Feature extraction -------------------------------------------------------
FRAMES_PER_VIDEO  = 30      # Sampled frames per video for temporal stability

# -- Model --------------------------------------------------------------------
RANDOM_STATE  = 42