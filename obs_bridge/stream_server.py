"""ZeypherLive — HTTP MJPEG Stream Server for mobile/browser viewing"""
import cv2
import numpy as np
import threading
import time
import io
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from config.settings import CONFIG


class StreamServer:
    def __init__(self, config=None):
        self.config = config or CONFIG.cloud
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._clients = 0
        self._running = False
        self._source = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()

    def set_source(self, source):
        self._source = source

    def start(self) -> bool:
        if self._running:
            return True
        try:
            handler = self._make_handler()
            self._server = HTTPServer(("0.0.0.0", self.config.server_port), handler)
            self._running = True
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            print(f"[Stream] Started on port {self.config.server_port}")
            return True
        except Exception as e:
            print(f"[Stream] Failed: {e}")
            return False

    def _make_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    server_ref._clients += 1
                    try:
                        while server_ref._running:
                            frame = server_ref._read_frame()
                            if frame is None:
                                time.sleep(0.03)
                                continue
                            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            data = jpeg.tobytes()
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        server_ref._clients -= 1

                elif self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    html = self._build_page()
                    self.wfile.write(html.encode())

                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    import json
                    status = {
                        "clients": server_ref._clients,
                        "fps": round(server_ref._fps, 1),
                        "frames": server_ref._frame_count,
                    }
                    self.wfile.write(json.dumps(status).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def _build_page(self):
                return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>ZeypherLive</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; color:#e0e0e0; font-family:system-ui; display:flex; flex-direction:column; align-items:center; min-height:100vh; }}
h1 {{ font-size:18px; padding:12px; color:#533483; }}
.video-wrap {{ position:relative; width:100%; max-width:640px; }}
img {{ width:100%; display:block; border-radius:8px; }}
.status {{ position:absolute; top:8px; left:8px; background:rgba(0,0,0,0.7); padding:4px 10px; border-radius:4px; font-size:12px; color:#55ff55; }}
.info {{ padding:16px; text-align:center; font-size:13px; color:#888; }}
.controls {{ padding:12px; display:flex; gap:8px; }}
.controls button {{ background:#533483; color:white; border:none; padding:10px 20px; border-radius:6px; font-size:14px; cursor:pointer; }}
.controls button:active {{ background:#6a42a0; }}
</style>
</head>
<body>
<h1>ZeypherLive</h1>
<div class="video-wrap">
  <img src="/stream" id="stream">
  <div class="status" id="status">Connecting...</div>
</div>
<div class="controls">
  <button onclick="toggleFullscreen()">Fullscreen</button>
</div>
<div class="info">AI output stream — works on any device on your network</div>
<script>
function toggleFullscreen() {{
  if (!document.fullscreenElement) {{
    document.documentElement.requestFullscreen();
  }} else {{
    document.exitFullscreen();
  }}
}}
setInterval(() => {{
  fetch("/status").then(r=>r.json()).then(d => {{
    document.getElementById("status").textContent = d.fps + " FPS | " + d.clients + " viewers";
  }}).catch(() => {{
    document.getElementById("status").textContent = "Disconnected";
  }});
}}, 1000);
</script>
</body>
</html>"""

            def log_message(self, format, *args):
                pass

        return Handler

    def _read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._source is not None and hasattr(self._source, 'read'):
                frame = self._source.read()
                if frame is not None:
                    self._frame = frame
                    self._frame_count += 1
                    now = time.time()
                    elapsed = now - self._last_fps_time
                    if elapsed >= 1.0:
                        self._fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._last_fps_time = now
                    return frame
            return self._frame

    def push_frame(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def _serve(self):
        self._server.serve_forever()

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._clients = 0
        print("[Stream] Stopped")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def url(self) -> str:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "127.0.0.1"
        return f"http://{ip}:{self.config.server_port}"

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "clients": self._clients,
            "fps": round(self._fps, 1),
            "url": self.url if self._running else "",
        }
