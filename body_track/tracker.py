"""ZeypherLive — Real Body Movement Overlay (Pose Mesh + Segmentation)"""
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

BODY_TRIANGLES = [
    (11, 12, 23), (12, 23, 24),
    (11, 13, 15), (12, 14, 16),
    (23, 25, 27), (24, 26, 28),
    (25, 27, 29), (26, 28, 30),
    (15, 17, 19), (16, 18, 20),
    (15, 19, 21), (16, 20, 22),
    (29, 31, 27), (30, 32, 28),
]

MESH_COLORS = [
    (53, 83, 255), (83, 53, 255), (128, 53, 255),
    (170, 53, 255), (200, 83, 255), (255, 83, 170),
    (255, 53, 128), (255, 53, 83), (255, 83, 53),
    (255, 128, 53), (255, 170, 53), (255, 200, 53),
]


class BodyTracker:
    def __init__(self, config=None):
        self.config = config or CONFIG.body_track
        self.landmarker = None
        self.last_result: Optional[PoseResult] = None
        self._prev_landmarks = None
        self._smooth_factor = 0.6
        self._mesh_opacity = 0.35
        self._glow_enabled = True
        self._trail_points = []
        self._max_trail = 20
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
                print("[BodyTracker] Model not found")
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
            print("[BodyTracker] Pose landmarker loaded")
        except Exception as e:
            print(f"[BodyTracker] Init failed: {e}")
            self.landmarker = None

    def process(self, frame: np.ndarray, timestamp: float = None) -> Optional[PoseResult]:
        if timestamp is None:
            timestamp = time.time()
        if self.landmarker is None:
            return None
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

                raw_lm = results.pose_landmarks[0].landmark
                if self._prev_landmarks and len(self._prev_landmarks) == len(raw_lm):
                    smoothed = []
                    for i, lm in enumerate(raw_lm):
                        prev = self._prev_landmarks[i]
                        sx = prev.x * self._smooth_factor + lm.x * (1 - self._smooth_factor)
                        sy = prev.y * self._smooth_factor + lm.y * (1 - self._smooth_factor)
                        sz = prev.z * self._smooth_factor + lm.z * (1 - self._smooth_factor)
                        smoothed.append(type(lm)(x=sx, y=sy, z=sz, visibility=lm.visibility))
                    raw_lm = smoothed
                self._prev_landmarks = raw_lm

                self.last_result = PoseResult(
                    landmarks=raw_lm,
                    world_landmarks=world_lm,
                    segmentation_mask=seg_mask,
                    timestamp=timestamp,
                )
                return self.last_result
        except Exception as e:
            pass
        return None

    def apply_body_overlay(self, frame: np.ndarray, result: PoseResult = None, mode: str = "mesh") -> np.ndarray:
        if result is None:
            result = self.last_result
        if result is None:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        if mode == "mesh":
            self._draw_body_mesh(overlay, result, h, w)
        elif mode == "glow":
            self._draw_glow_body(overlay, result, h, w)
        elif mode == "outline":
            self._draw_body_outline(overlay, result, h, w)
        elif mode == "particles":
            self._draw_body_particles(overlay, result, h, w)

        if self._glow_enabled and mode == "mesh":
            glow = cv2.GaussianBlur(overlay, (0, 0), 15)
            overlay = cv2.addWeighted(overlay, 0.7, glow, 0.3, 0)

        return cv2.addWeighted(frame, 1 - self._mesh_opacity, overlay, self._mesh_opacity, 0)

    def _draw_body_mesh(self, overlay: np.ndarray, result: PoseResult, h: int, w: int):
        lm = result.landmarks
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        for i, tri in enumerate(BODY_TRIANGLES):
            if tri[0] < len(pts) and tri[1] < len(pts) and tri[2] < len(pts):
                p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                color = MESH_COLORS[i % len(MESH_COLORS)]
                tri_pts = np.array([p1, p2, p3], dtype=np.int32)
                cv2.fillPoly(overlay, [tri_pts], color)

        for start_idx, end_idx in BODY_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(overlay, pts[start_idx], pts[end_idx], (255, 255, 255), 2, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(overlay, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_glow_body(self, overlay: np.ndarray, result: PoseResult, h: int, w: int):
        lm = result.landmarks
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        for start_idx, end_idx in BODY_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                p1, p2 = pts[start_idx], pts[end_idx]
                cv2.line(overlay, p1, p2, (0, 200, 255), 6, cv2.LINE_AA)
                cv2.line(overlay, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(overlay, pt, 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_body_outline(self, overlay: np.ndarray, result: PoseResult, h: int, w: int):
        lm = result.landmarks
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        outline_indices = [11, 12, 24, 23, 11]
        outline_pts = [pts[i] for i in outline_indices if i < len(pts)]
        if len(outline_pts) >= 3:
            cv2.polylines(overlay, [np.array(outline_pts)], True, (0, 255, 128), 3, cv2.LINE_AA)

        for arm_side in [(11, 13, 15), (12, 14, 16)]:
            arm_pts = [pts[i] for i in arm_side if i < len(pts)]
            if len(arm_pts) >= 2:
                cv2.polylines(overlay, [np.array(arm_pts)], False, (0, 255, 128), 2, cv2.LINE_AA)

        for leg_side in [(23, 25, 27, 29, 31), (24, 26, 28, 30, 32)]:
            leg_pts = [pts[i] for i in leg_side if i < len(pts)]
            if len(leg_pts) >= 2:
                cv2.polylines(overlay, [np.array(leg_pts)], False, (0, 255, 128), 2, cv2.LINE_AA)

    def _draw_body_particles(self, overlay: np.ndarray, result: PoseResult, h: int, w: int):
        lm = result.landmarks
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        for start_idx, end_idx in BODY_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                p1, p2 = pts[start_idx], pts[end_idx]
                length = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
                for j in range(0, max(length, 1), 4):
                    t = j / max(length, 1)
                    px = int(p1[0] + (p2[0]-p1[0]) * t)
                    py = int(p1[1] + (p2[1]-p1[1]) * t)
                    np.random.seed(px * 1000 + py)
                    r = np.random.randint(2, 5)
                    color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                    cv2.circle(overlay, (px, py), r, color, -1, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(overlay, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_landmarks(self, frame: np.ndarray, result: PoseResult = None) -> np.ndarray:
        if result is None:
            result = self.last_result
        if result is None:
            return frame
        annotated = frame.copy()
        h, w = frame.shape[:2]
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
        self._prev_landmarks = None
        self._trail_points = []
        self._init_pose()

    def __del__(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
