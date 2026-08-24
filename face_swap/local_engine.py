"""ZeypherLive — Local Face Swap Engine (OpenCV + MediaPipe Tasks)"""
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from typing import Optional, NamedTuple


class FaceInfo(NamedTuple):
    bbox: tuple
    confidence: float
    landmarks: Optional[np.ndarray] = None


class LocalFaceSwap:
    def __init__(self):
        self._source_face = None
        self._source_bbox = None
        self._mp_detector = None
        self._mp_landmarker = None
        self._haar = None
        self._initialized = False
        self._models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self._method = "none"

    def initialize(self) -> bool:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model_path = os.path.join(self._models_dir, "blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.expanduser("~"), "Desktop", "ZeypherLive", "models", "blaze_face_short_range.tflite")
            if os.path.exists(model_path):
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.5,
                )
                self._mp_detector = vision.FaceDetector.create_from_options(options)
                self._method = "mediapipe"
                print("[LocalSwap] MediaPipe FaceDetector loaded")
                self._initialized = True

                try:
                    mesh_model = os.path.join(self._models_dir, "face_landmarker.task")
                    if not os.path.exists(mesh_model):
                        mesh_model = os.path.join(os.path.expanduser("~"), "Desktop", "ZeypherLive", "models", "face_landmarker.task")
                    if os.path.exists(mesh_model):
                        base_options2 = mp_python.BaseOptions(model_asset_path=mesh_model)
                        mesh_options = vision.FaceLandmarkerOptions(
                            base_options=base_options2,
                            running_mode=vision.RunningMode.VIDEO,
                            num_faces=1,
                            min_face_detection_confidence=0.3,
                            min_tracking_confidence=0.3,
                        )
                        self._mp_mesh = vision.FaceLandmarker.create_from_options(mesh_options)
                        print("[LocalSwap] FaceLandmarker loaded for lip sync")
                except Exception as e:
                    print(f"[LocalSwap] FaceLandmarker not available: {e}")

                return True
        except Exception as e:
            print(f"[LocalSwap] MediaPipe init: {e}")

        try:
            cascade_path = os.path.join(self._models_dir, "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if not self._haar.empty():
                self._method = "haar"
                print("[LocalSwap] Haar cascade loaded")
                self._initialized = True
                return True
        except Exception:
            pass

        print("[LocalSwap] No face detection available")
        return False

    def set_source_face(self, image_path: str) -> bool:
        if not self._initialized:
            return False
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            faces = self._detect(img)
            if len(faces) > 0:
                best = max(faces, key=lambda f: f.confidence)
                x1, y1, x2, y2 = best.bbox
                margin = int(0.2 * max(x2 - x1, y2 - y1))
                h, w = img.shape[:2]
                sx1 = max(0, x1 - margin)
                sy1 = max(0, y1 - margin)
                sx2 = min(w, x2 + margin)
                sy2 = min(h, y2 + margin)
                self._source_face = img[sy1:sy2, sx1:sx2].copy()
                self._source_bbox = (sx1, sy1, sx2, sy2)
                print(f"[LocalSwap] Source face set: {self._source_face.shape}")
                return True
            print("[LocalSwap] No face in source image")
            return False
        except Exception as e:
            print(f"[LocalSwap] Set source error: {e}")
            return False

    def set_source_from_frame(self, frame: np.ndarray) -> bool:
        if not self._initialized:
            return False
        try:
            faces = self._detect(frame)
            if len(faces) > 0:
                best = max(faces, key=lambda f: f.confidence)
                x1, y1, x2, y2 = best.bbox
                margin = int(0.2 * max(x2 - x1, y2 - y1))
                h, w = frame.shape[:2]
                sx1 = max(0, x1 - margin)
                sy1 = max(0, y1 - margin)
                sx2 = min(w, x2 + margin)
                sy2 = min(h, y2 + margin)
                self._source_face = frame[sy1:sy2, sx1:sx2].copy()
                self._source_bbox = (sx1, sy1, sx2, sy2)
                print(f"[LocalSwap] Source face set from frame: {self._source_face.shape}")
                return True
            print("[LocalSwap] No face detected in frame")
            return False
        except Exception as e:
            print(f"[LocalSwap] Set source frame error: {e}")
            return False

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self._initialized or self._source_face is None:
            return None
        try:
            faces = self._detect(frame)
            if not faces:
                return frame
            result = frame.copy()
            for face in faces:
                result = self._swap_face(result, face.bbox)
            return result
        except Exception as e:
            print(f"[LocalSwap] Process error: {e}")
            return frame

    def _detect(self, img: np.ndarray) -> list:
        faces = []

        if self._mp_detector is not None:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self._mp_detector.detect(mp_image)
                if results.detections:
                    h, w = img.shape[:2]
                    for det in results.detections:
                        bb = det.bounding_box
                        x1 = max(0, bb.origin_x)
                        y1 = max(0, bb.origin_y)
                        x2 = min(w, bb.origin_x + bb.width)
                        y2 = min(h, bb.origin_y + bb.height)
                        if x2 > x1 and y2 > y1:
                            faces.append(FaceInfo(
                                bbox=(x1, y1, x2, y2),
                                confidence=det.categories[0].score if det.categories else 0.9,
                            ))
                return faces
            except Exception as e:
                print(f"[LocalSwap] MediaPipe detect error: {e}")

        if self._haar is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = self._haar.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                for (x, y, w, h) in rects:
                    margin = int(0.2 * max(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(img.shape[1], x + w + margin)
                    y2 = min(img.shape[0], y + h + margin)
                    faces.append(FaceInfo(bbox=(x1, y1, x2, y2), confidence=0.8))
            except Exception:
                pass

        return faces

    def _swap_face(self, frame: np.ndarray, target_bbox: tuple) -> np.ndarray:
        if self._source_face is None:
            return frame
        x1, y1, x2, y2 = target_bbox
        th = y2 - y1
        tw = x2 - x1
        if th <= 0 or tw <= 0:
            return frame

        src_resized = cv2.resize(self._source_face, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask = 255 * np.ones((th, tw, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        cx = max(tw // 2, min(w - tw // 2, x1 + tw // 2))
        cy = max(th // 2, min(h - th // 2, y1 + th // 2))

        try:
            output = cv2.seamlessClone(src_resized, frame, mask, (cx, cy), cv2.NORMAL_CLONE)
            return output
        except Exception:
            output = frame.copy()
            output[y1:y2, x1:x2] = src_resized
            return output

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def has_source(self) -> bool:
        return self._source_face is not None

    @property
    def stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "has_source": self._source_face is not None,
            "method": self._method,
            "has_mesh": self._mp_mesh is not None,
        }

    def get_face_landmarks(self, frame: np.ndarray):
        if self._mp_mesh is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            results = self._mp_mesh.detect_for_video(mp_image, ts_ms)
            if results.face_landmarks:
                return results.face_landmarks[0].landmark
        except Exception:
            pass
        return None

    def __del__(self):
        if self._mp_detector:
            try:
                self._mp_detector.close()
            except Exception:
                pass
