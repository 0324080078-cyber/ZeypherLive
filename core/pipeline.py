"""ZeypherLive Core — Frame Processing Pipeline"""
import cv2
import numpy as np
import time
import threading
from typing import Optional, Protocol
from config.settings import CONFIG


class FrameProcessor(Protocol):
    def process(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]: ...


class FramePipeline:
    def __init__(self):
        self.processors: list[FrameProcessor] = []
        self.running = False
        self.output_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self._frame_count = 0
        self._last_time = time.time()
        self._callbacks: list = []
        self._thread: Optional[threading.Thread] = None
        self._source = None

    def set_source(self, source):
        self._source = source

    def add_processor(self, processor: FrameProcessor):
        self.processors.append(processor)

    def remove_processor(self, processor: FrameProcessor):
        if processor in self.processors:
            self.processors.remove(processor)

    def register_output_callback(self, callback):
        self._callbacks.append(callback)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._thread.start()

    def _pipeline_loop(self):
        while self.running:
            if self._source is None:
                time.sleep(0.01)
                continue
            frame = self._source.read()
            if frame is None:
                time.sleep(0.005)
                continue
            timestamp = time.time()
            for processor in self.processors:
                try:
                    result = processor.process(frame, timestamp)
                    if result is not None:
                        frame = result
                except Exception as e:
                    pass
            with self.lock:
                self.output_frame = frame
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_time
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_time = now
            for cb in self._callbacks:
                try:
                    cb(frame)
                except Exception:
                    pass

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.output_frame.copy() if self.output_frame is not None else None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def process_single(self, frame: np.ndarray) -> Optional[np.ndarray]:
        timestamp = time.time()
        for processor in self.processors:
            try:
                result = processor.process(frame, timestamp)
                if result is not None:
                    frame = result
            except Exception:
                pass
        return frame
