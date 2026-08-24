"""ZeypherLive — Face Swap Engine (InsightFace + Lucy Hybrid)"""
import cv2
import numpy as np
import os
import time
from typing import Optional, NamedTuple
from config.settings import CONFIG


class FaceSwapResult(NamedTuple):
    swapped_frame: np.ndarray
    face_count: int
    swap_latency_ms: float
    method: str


class FaceAnalyzer:
    def __init__(self):
        self._det_model = None
        self._swapper_model = None
        self._initialized = False

    def initialize(self, model_dir: str = None):
        if self._initialized:
            return True
        try:
            import onnxruntime as ort
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            det_path = os.path.join(model_dir, "detection_10g.onnx")
            swapper_path = os.path.join(model_dir, "inswapper_128.onnx")
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            if os.path.exists(det_path):
                self._det_model = ort.InferenceSession(det_path, providers=providers)
            if os.path.exists(swapper_path):
                self._swapper_model = ort.InferenceSession(swapper_path, providers=providers)
            self._initialized = True
            return True
        except ImportError:
            return False

    def detect_faces(self, frame: np.ndarray) -> list:
        if not self._initialized or self._det_model is None:
            return self._detect_faces_opencv(frame)
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (640, 640), (104, 117, 123))
            self._det_model.run(None, {"input": blob})
            input_name = self._det_model.get_inputs()[0].name
            outputs = self._det_model.run(None, {input_name: blob})
            faces = []
            if len(outputs) > 0:
                detections = outputs[0]
                for det in detections:
                    if det[4] > 0.5:
                        x1, y1, x2, y2 = det[:4]
                        h, w = frame.shape[:2]
                        faces.append({
                            "bbox": (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                            "confidence": float(det[4]),
                        })
            return faces
        except Exception:
            return self._detect_faces_opencv(frame)

    def _detect_faces_opencv(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            "haarcascade_frontalface_default.xml",
        ]
        face_cascade = None
        for p in cascade_paths:
            if os.path.exists(p):
                face_cascade = cv2.CascadeClassifier(p)
                if not face_cascade.empty():
                    break
                face_cascade = None
        if face_cascade is None:
            return self._detect_faces_dnn(frame)
        faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        faces = []
        for (x, y, w, h) in faces_rect:
            margin = int(0.2 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.9,
            })
        return faces

    def _detect_faces_dnn(self, frame: np.ndarray) -> list:
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net = cv2.dnn.readNetFromCaffe(
                os.path.join(os.path.dirname(__file__), "..", "models", "deploy.prototxt"),
                os.path.join(os.path.dirname(__file__), "..", "models", "res10_300x300_ssd_iter_140000.caffemodel"),
            )
            net.setInput(blob)
            detections = net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    faces.append({"bbox": (x1, y1, x2, y2), "confidence": float(confidence)})
            return faces
        except Exception:
            return []

    def get_face_embedding(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        face_resized = cv2.resize(face_img, (112, 112))
        blob = face_resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        if self._det_model is not None:
            try:
                input_name = self._det_model.get_inputs()[0].name
                embedding = self._det_model.run(None, {input_name: blob})[0]
                return embedding.flatten()
            except Exception:
                pass
        return np.random.randn(512).astype(np.float32)


class FaceSwapEngine:
    def __init__(self, config=None):
        self.config = config or CONFIG.face_swap
        self.analyzer = FaceAnalyzer()
        self._source_embedding: Optional[np.ndarray] = None
        self._source_face: Optional[np.ndarray] = None
        self._initialized = False
        self._lucy_client = None

    def initialize(self):
        if self._initialized:
            return True
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self._initialized = self.analyzer.initialize(model_dir)
        return self._initialized

    def set_source_face(self, image_path: str) -> bool:
        img = cv2.imread(image_path)
        if img is None:
            return False
        self._source_face = img.copy()
        faces = self.analyzer.detect_faces(img)
        if faces:
            self._source_embedding = self.analyzer.get_face_embedding(img, faces[0]["bbox"])
            return self._source_embedding is not None
        return False

    def set_source_from_frame(self, frame: np.ndarray) -> bool:
        self._source_face = frame.copy()
        faces = self.analyzer.detect_faces(frame)
        if faces:
            self._source_embedding = self.analyzer.get_face_embedding(frame, faces[0]["bbox"])
            return self._source_embedding is not None
        return False

    def set_lucy_client(self, client):
        self._lucy_client = client

    def swap_local(self, frame: np.ndarray, target_bbox: tuple) -> Optional[np.ndarray]:
        if self._source_face is None:
            return None
        x1, y1, x2, y2 = target_bbox
        src_h, src_w = self._source_face.shape[:2]
        target_w = x2 - x1
        target_h = y2 - y1
        source_resized = cv2.resize(self._source_face, (target_w, target_h))
        mask = np.ones((target_h, target_w, 3), dtype=np.float32) * 255
        center = (x1 + target_w // 2, y1 + target_h // 2)
        try:
            seamless = cv2.seamlessClone(
                source_resized, frame, mask, center, cv2.NORMAL_CLONE
            )
            return seamless
        except Exception:
            output = frame.copy()
            output[y1:y2, x1:x2] = source_resized
            return output

    def process(self, frame: np.ndarray, timestamp: float = None) -> Optional[np.ndarray]:
        if timestamp is None:
            timestamp = time.time()
        if self._source_embedding is None and self._source_face is None:
            return None
        faces = self.analyzer.detect_faces(frame)
        if not faces:
            return None
        if self.config.swap_method == "lucy" and self._lucy_client and self._lucy_client.connected:
            ref_b64 = self._lucy_client.set_reference_from_frame(self._source_face)
            self._lucy_client.send_frame_sync(
                frame,
                prompt="Substitute the character in the video with the reference image.",
                reference_image=ref_b64,
            )
            result = self._lucy_client.read()
            if result is not None:
                return result
        output = frame.copy()
        for face_info in faces:
            bbox = face_info["bbox"]
            swapped = self.swap_local(output, bbox)
            if swapped is not None:
                output = swapped
        return output

    def extract_all_faces(self, frame: np.ndarray) -> list:
        faces = self.analyzer.detect_faces(frame)
        extracted = []
        for fi in faces:
            x1, y1, x2, y2 = fi["bbox"]
            face_img = frame[y1:y2, x1:x2].copy()
            embedding = self.analyzer.get_face_embedding(frame, fi["bbox"])
            extracted.append({
                "image": face_img,
                "bbox": fi["bbox"],
                "confidence": fi["confidence"],
                "embedding": embedding,
            })
        return extracted

    def swap_faces_bidirectional(self, frame: np.ndarray, face_a_idx: int, face_b_idx: int) -> Optional[np.ndarray]:
        faces = self.analyzer.detect_faces(frame)
        if len(faces) < 2:
            return None
        emb_a = self.analyzer.get_face_embedding(frame, faces[face_a_idx]["bbox"])
        emb_b = self.analyzer.get_face_embedding(frame, faces[face_b_idx]["bbox"])
        if emb_a is None or emb_b is None:
            return None
        output = frame.copy()
        face_a_img = self._crop_face(frame, faces[face_a_idx]["bbox"])
        face_b_img = self._crop_face(frame, faces[face_b_idx]["bbox"])
        if face_a_img is not None and face_b_img is not None:
            output = self._paste_face(output, face_b_img, faces[face_a_idx]["bbox"])
            output = self._paste_face(output, face_a_img, faces[face_b_idx]["bbox"])
        return output

    def _crop_face(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _paste_face(self, frame: np.ndarray, face_img: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        target_w = x2 - x1
        target_h = y2 - y1
        if target_w <= 0 or target_h <= 0:
            return frame
        resized = cv2.resize(face_img, (target_w, target_h))
        output = frame.copy()
        try:
            mask = np.ones((target_h, target_w, 3), dtype=np.float32) * 255
            center = (x1 + target_w // 2, y1 + target_h // 2)
            output = cv2.seamlessClone(resized, output, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            output[y1:y2, x1:x2] = resized
        return output

    @property
    def is_ready(self) -> bool:
        return self._initialized and (self._source_embedding is not None or self._source_face is not None)
