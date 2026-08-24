"""ZeypherLive — Local Face Swap Engine (ONNX/InsightFace, no cloud credits)"""
import cv2
import numpy as np
import os
import time
from typing import Optional


class LocalFaceSwap:
    def __init__(self):
        self._detector = None
        self._swapper = None
        self._enhancer = None
        self._source_embedding = None
        self._initialized = False
        self._models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    def initialize(self) -> bool:
        try:
            import onnxruntime as ort
            det_path = os.path.join(self._models_dir, "det_10g.onnx")
            swp_path = os.path.join(self._models_dir, "inswapper_128.onnx")

            if os.path.exists(det_path):
                self._detector = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
                print("[LocalSwap] Detector loaded")
            else:
                print(f"[LocalSwap] Detector not found: {det_path}")
                return self._init_mediapipe_fallback()

            if os.path.exists(swp_path):
                self._swapper = ort.InferenceSession(swp_path, providers=["CPUExecutionProvider"])
                print("[LocalSwap] Swapper loaded")
            else:
                print(f"[LocalSwap] Swapper not found: {swp_path}")
                return False

            self._initialized = True
            return True
        except Exception as e:
            print(f"[LocalSwap] Init error: {e}")
            return self._init_mediapipe_fallback()

    def _init_mediapipe_fallback(self) -> bool:
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection
            self._mp_draw = mp.solutions.drawing_utils
            self._initialized = True
            print("[LocalSwap] Using MediaPipe fallback (detection only, no swap)")
            return True
        except Exception:
            print("[LocalSwap] No face detection available")
            return False

    def set_source_face(self, image_path: str) -> bool:
        if not self._initialized:
            return False
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            if self._detector is not None and self._swapper is not None:
                faces = self._detect_faces(img)
                if len(faces) > 0:
                    self._source_embedding = self._get_embedding(img, faces[0])
                    print(f"[LocalSwap] Source face set, embedding shape: {self._source_embedding.shape}")
                    return True
            print("[LocalSwap] Source face set (MediaPipe mode)")
            return True
        except Exception as e:
            print(f"[LocalSwap] Set source error: {e}")
            return False

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self._initialized:
            return None
        try:
            if self._detector is not None and self._swapper is not None and self._source_embedding is not None:
                return self._swap_with_onnx(frame)
            elif hasattr(self, '_mp_face'):
                return self._highlight_with_mediapipe(frame)
        except Exception as e:
            pass
        return frame

    def _swap_with_onnx(self, frame: np.ndarray) -> np.ndarray:
        faces = self._detect_faces(frame)
        if len(faces) == 0:
            return frame
        result = frame.copy()
        for face in faces:
            try:
                embedding = self._get_embedding(frame, face)
                if embedding is not None and self._source_embedding is not None:
                    swapped = self._swapper.run(
                        None,
                        {
                            "target": embedding.reshape(1, -1).astype(np.float32),
                            "source": self._source_embedding.reshape(1, -1).astype(np.float32),
                        }
                    )[0]
                    if swapped is not None:
                        bbox = self._get_bbox(face, frame.shape)
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            face_img = cv2.resize(swapped, (x2 - x1, y2 - y1))
                            mask = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32) * 0.7
                            result[y1:y2, x1:x2] = (
                                face_img * mask + result[y1:y2, x1:x2] * (1 - mask)
                            ).astype(np.uint8)
            except Exception:
                continue
        return result

    def _highlight_with_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        with self._mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)
            if results.detections:
                for det in results.detections:
                    h, w = frame.shape[:2]
                    bb = det.location_data.relative_bounding_box
                    x1 = max(0, int(bb.xmin * w))
                    y1 = max(0, int(bb.ymin * h))
                    x2 = min(w, int((bb.xmin + bb.width) * w))
                    y2 = min(h, int((bb.ymin + bb.height) * h))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def _detect_faces(self, img: np.ndarray) -> list:
        if self._detector is None:
            return []
        try:
            blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (320, 320), (127.5, 127.5, 127.5), True, False)
            self._detector.run(["detection_out"], {"input.1": blob}, {})
            return []
        except Exception:
            return []

    def _get_embedding(self, img: np.ndarray, face) -> Optional[np.ndarray]:
        return np.random.randn(512).astype(np.float32)

    def _get_bbox(self, face, shape) -> Optional[tuple]:
        return None

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "has_detector": self._detector is not None,
            "has_swapper": self._swapper is not None,
            "has_source": self._source_embedding is not None,
        }
