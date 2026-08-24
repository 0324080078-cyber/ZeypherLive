"""ZeypherLive — MediaPipe Body Pose Tracker (v2 — Tasks API)"""
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from typing import Optional, NamedTuple
from config.settings import CONFIG


class PoseResult(NamedTuple):
    landmarks: list
    world_landmarks: list
    segmentation_mask: Optional[np.ndarray]
    timestamp: float


BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


class BodyTracker:
    def __init__(self, config=None):
        self.config = config or CONFIG.body_track
        self.landmarker = None
        self.last_result: Optional[PoseResult] = None
        self._base_options = None
        self._init_pose()

    def _init_pose(self):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "pose_landmarker_heavy.task"
            )
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    os.path.expanduser("~"), "Desktop", "ZeypherLive", "models", "pose_landmarker_heavy.task"
                )
            if not os.path.exists(model_path):
                print("[BodyTracker] Model not found, using fallback pose detection")
                self.landmarker = None
                return
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.config.min_detection_confidence,
                min_pose_presence_confidence=self.config.min_tracking_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=self.config.enable_segmentation,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[BodyTracker] Init failed: {e}")
            self.landmarker = None

    def process(self, frame: np.ndarray, timestamp: float = None) -> Optional[PoseResult]:
        if timestamp is None:
            timestamp = time.time()
        if self.landmarker is None:
            return self._fallback_process(frame, timestamp)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(timestamp * 1000)
            results = self.landmarker.detect_for_video(mp_image, ts_ms)
            if results.pose_landmarks:
                seg_mask = None
                if results.segmentation_masks:
                    seg_mask = results.segmentation_masks[0].numpy_view()
                world_lm = None
                if results.pose_world_landmarks:
                    world_lm = results.pose_world_landmarks[0].landmark
                self.last_result = PoseResult(
                    landmarks=results.pose_landmarks[0].landmark,
                    world_landmarks=world_lm,
                    segmentation_mask=seg_mask,
                    timestamp=timestamp,
                )
                return self.last_result
        except Exception as e:
            pass
        return None

    def _fallback_process(self, frame: np.ndarray, timestamp: float) -> Optional[PoseResult]:
        return None

    def draw_landmarks(self, frame: np.ndarray, result: PoseResult = None) -> np.ndarray:
        if result is None:
            result = self.last_result
        if result is None:
            return frame
        annotated = frame.copy()
        h, w = frame.shape[:2]
        if self.config.show_skeleton:
            for lm in result.landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
            for start_idx, end_idx in BODY_CONNECTIONS:
                if start_idx < len(result.landmarks) and end_idx < len(result.landmarks):
                    p1 = (int(result.landmarks[start_idx].x * w), int(result.landmarks[start_idx].y * h))
                    p2 = (int(result.landmarks[end_idx].x * w), int(result.landmarks[end_idx].y * h))
                    cv2.line(annotated, p1, p2, (0, 255, 255), 2)
        return annotated

    def get_body_bbox(self, result: PoseResult = None) -> Optional[tuple]:
        if result is None:
            result = self.last_result
        if result is None or not result.landmarks:
            return None
        xs = [lm.x for lm in result.landmarks]
        ys = [lm.y for lm in result.landmarks]
        return (min(xs), min(ys), max(xs), max(ys))

    def extract_keypoints(self, result: PoseResult = None) -> np.ndarray:
        if result is None:
            result = self.last_result
        if result is None:
            return np.array([])
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.landmarks])

    def reset(self):
        self.last_result = None
        self._init_pose()

    def __del__(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
