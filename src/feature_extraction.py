"""
feature_extraction.py
=====================
Implements behavioral feature extraction for Prakriti classification.
Uses MediaPipe Tasks API (v1.1+) for landmark detection.
Focuses on temporal dynamics:
  1. Hand Gestures: Variance of landmark velocity.
  2. Head Movements: Head pose rotation frequency.
  3. Stage Movement: Path Tortuosity.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode
)
from config import *


def calculate_velocity_variance(landmarks_series):
    """
    Calculates the variance of the velocity of landmarks over time.
    Higher variance = more erratic/irregular movements (Vata indicator).
    """
    if len(landmarks_series) < 2:
        return 0.0
    velocities = np.diff(landmarks_series, axis=0)
    mag = np.linalg.norm(velocities, axis=2)
    frame_vel = np.mean(mag, axis=1)
    return float(np.var(frame_vel))


def calculate_head_rotation_frequency(face_landmarks_series):
    """
    Tracks head pose rotation frequency via orientation vector variance.
    Higher frequency = more intense/deliberate head changes (Pitta indicator).
    """
    if len(face_landmarks_series) < 2:
        return 0.0
    orientations = []
    for frame in face_landmarks_series:
        if len(frame) < 264:
            continue
        eye_mid = (np.array(frame[33]) + np.array(frame[263])) / 2
        nose = np.array(frame[1])
        vec = nose - eye_mid
        orientations.append(vec / (np.linalg.norm(vec) + 1e-6))
    if not orientations:
        return 0.0
    orientations = np.array(orientations)
    return float(np.var(orientations, axis=0).sum())


def calculate_path_tortuosity(pose_centroids):
    """
    Calculates Path Tortuosity = Total Path Length / Net Displacement.
    A value of 1.0 means perfectly straight movement.
    Higher values indicate more wandering/unpredictable movement.
    """
    if len(pose_centroids) < 2:
        return 1.0
    centroids = np.array(pose_centroids)
    diffs = np.diff(centroids, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))
    displacement = np.linalg.norm(centroids[-1] - centroids[0])
    if displacement < 1e-6:
        return 1.0
    return float(path_length / displacement)


def extract_behavioral_features(video_path):
    """
    Extracts the 'Focus Three' behavioral features from a single video
    using the MediaPipe Tasks API (v1.1+).
    Returns: np.array([hand_velocity_var, head_rotation_freq, path_tortuosity])
    """
    # Verify model files exist
    for mpath in [HAND_MODEL_PATH, FACE_MODEL_PATH, POSE_MODEL_PATH]:
        if not os.path.exists(mpath):
            print(f"[ERROR] Model file not found: {mpath}")
            return np.zeros(3, dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = FRAMES_PER_VIDEO
    step = max(1, total_frames // target_frames) if total_frames > 0 else 1

    # Create landmarkers using Tasks API
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5
    )

    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    face_landmarker = FaceLandmarker.create_from_options(face_options)
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    hand_landmarks_list = []
    face_landmarks_list = []
    pose_centroids = []

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    try:
        while cap.isOpened() and len(hand_landmarks_list) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_count * 1000.0 / fps)

                # 1. Hand Tracking
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                    lm = hand_result.hand_landmarks[0]
                    hand_landmarks_list.append(np.array([[l.x, l.y, l.z] for l in lm]))
                else:
                    hand_landmarks_list.append(
                        hand_landmarks_list[-1] if hand_landmarks_list else np.zeros((21, 3))
                    )

                # 2. Face Tracking
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                    lm = face_result.face_landmarks[0]
                    face_landmarks_list.append(np.array([[l.x, l.y, l.z] for l in lm]))
                else:
                    face_landmarks_list.append(
                        face_landmarks_list[-1] if face_landmarks_list else np.zeros((478, 3))
                    )

                # 3. Pose Tracking (centroid = midpoint of hips)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    lm = pose_result.pose_landmarks[0]
                    centroid = (np.array([lm[23].x, lm[23].y]) + np.array([lm[24].x, lm[24].y])) / 2
                    pose_centroids.append(centroid)
                else:
                    pose_centroids.append(
                        pose_centroids[-1] if pose_centroids else np.array([0.5, 0.5])
                    )

            frame_count += 1
    finally:
        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()
        cap.release()

    if not hand_landmarks_list or not face_landmarks_list or not pose_centroids:
        return np.zeros(3, dtype=np.float32)

    return np.array([
        calculate_velocity_variance(hand_landmarks_list),
        calculate_head_rotation_frequency(face_landmarks_list),
        calculate_path_tortuosity(pose_centroids)
    ], dtype=np.float32)


def build_behavioral_features(filenames, video_dir):
    """
    Processes all videos and returns (N, 3) feature array.
    Uses tqdm for live progress tracking.
    """
    features = []
    skipped = 0
    for fn in tqdm(filenames, desc="Extracting Behavioral Features", unit="video"):
        vpath = os.path.join(video_dir, fn)
        if not os.path.exists(vpath):
            features.append(np.zeros(3, dtype=np.float32))
            skipped += 1
            continue
        feat = extract_behavioral_features(vpath)
        features.append(feat if feat is not None else np.zeros(3, dtype=np.float32))

    if skipped > 0:
        print(f"  [WARNING] {skipped}/{len(filenames)} videos not found (zero-features used).")

    return np.array(features, dtype=np.float32)