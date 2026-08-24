"""ZeypherLive — Mobile-Optimized MJPEG Stream Server"""
import cv2
import numpy as np
import threading
import time
import json
import socket
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
        self._audio_clients = 0
        self._audio_stream = None

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
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
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
                            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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

                elif self.path == "/snapshot":
                    frame = server_ref._read_frame()
                    if frame is not None:
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                    else:
                        self.send_response(503)
                        self.end_headers()

                elif self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(self._build_mobile_page().encode())

                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    status = {
                        "clients": server_ref._clients,
                        "fps": round(server_ref._fps, 1),
                        "frames": server_ref._frame_count,
                        "uptime": round(time.time() - server_ref._start_time, 0) if hasattr(server_ref, '_start_time') else 0,
                    }
                    self.wfile.write(json.dumps(status).encode())

                elif self.path == "/control":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(self._build_control_page().encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def _build_mobile_page(self):
                return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0a0a0a">
<title>ZeypherLive</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#000;font-family:-apple-system,system-ui,sans-serif}
#stream{width:100%;height:100%;object-fit:contain;display:block}
.overlay{position:fixed;top:0;left:0;right:0;padding:8px 12px;display:flex;justify-content:space-between;align-items:center;background:linear-gradient(180deg,rgba(0,0,0,0.7) 0%,transparent 100%);z-index:10}
.logo{font-size:14px;font-weight:700;color:#8b8bff}
.status-pill{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:12px;font-size:11px;font-weight:600}
.status-live{background:rgba(85,255,85,0.2);color:#55ff55}
.status-off{background:rgba(255,85,85,0.2);color:#ff5555}
.dot{width:6px;height:6px;border-radius:50%;animation:pulse 1.5s infinite}
.dot-live{background:#55ff55}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.bottom-bar{position:fixed;bottom:0;left:0;right:0;padding:12px;background:linear-gradient(0deg,rgba(0,0,0,0.8) 0%,transparent 100%);display:flex;justify-content:center;gap:10px;z-index:10}
.btn{padding:10px 18px;border:none;border-radius:25px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.2s}
.btn-primary{background:#533483;color:white}
.btn-primary:active{background:#6a42a0;transform:scale(0.95)}
.btn-outline{background:rgba(255,255,255,0.1);color:white;border:1px solid rgba(255,255,255,0.2)}
.btn-outline:active{background:rgba(255,255,255,0.2)}
.fps-badge{position:fixed;bottom:60px;right:12px;background:rgba(0,0,0,0.6);padding:4px 8px;border-radius:8px;font-size:11px;color:#888;z-index:10}
.loading{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:#888;font-size:14px;z-index:5}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{width:30px;height:30px;border:3px solid #333;border-top-color:#533483;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 10px}
</style>
</head>
<body>
<div class="overlay">
<div class="logo">ZeypherLive</div>
<div class="status-pill status-off" id="status"><div class="dot" id="dot"></div><span id="status-text">Connecting...</span></div>
</div>
<div class="loading" id="loading"><div class="spinner"></div>Loading stream...</div>
<img id="stream" src="/stream" onload="document.getElementById('loading').style.display='none'" onerror="retryStream()">
<div class="fps-badge" id="fps">-- FPS</div>
<div class="bottom-bar">
<button class="btn btn-primary" onclick="toggleFullscreen()">Fullscreen</button>
<button class="btn btn-outline" onclick="screenshot()">Screenshot</button>
<button class="btn btn-outline" onclick="toggleMute()">Mute</button>
</div>
<script>
let reconnectTimer;
function retryStream(){
    document.getElementById('loading').style.display='block';
    clearTimeout(reconnectTimer);
    reconnectTimer=setTimeout(()=>{
        document.getElementById('stream').src='/stream?'+Date.now();
    },2000);
}
document.getElementById('stream').onerror=retryStream;
function toggleFullscreen(){
    if(!document.fullscreenElement){
        document.documentElement.requestFullscreen().catch(()=>{});
    }else{
        document.exitFullscreen();
    }
}
function screenshot(){
    const c=document.createElement('canvas');
    const img=document.getElementById('stream');
    c.width=img.naturalWidth||640;
    c.height=img.naturalHeight||480;
    c.getContext('2d').drawImage(img,0,0);
    const a=document.createElement('a');
    a.href=c.toDataURL('image/png');
    a.download='zeypher_'+Date.now()+'.png';
    a.click();
}
let muted=false;
function toggleMute(){
    muted=!muted;
    document.querySelector('.btn-outline:last-child').textContent=muted?'Unmute':'Mute';
}
setInterval(()=>{
    fetch('/status').then(r=>r.json()).then(d=>{
        document.getElementById('fps').textContent=d.fps+' FPS';
        document.getElementById('status-text').textContent=d.fps>0?d.fps+' FPS | '+d.clients+' viewers':'Waiting...';
        const pill=document.getElementById('status');
        const dot=document.getElementById('dot');
        if(d.fps>0){pill.className='status-pill status-live';dot.className='dot dot-live';}
        else{pill.className='status-pill status-off';dot.className='dot';}
    }).catch(()=>{
        document.getElementById('status-text').textContent='Reconnecting...';
        document.getElementById('status').className='status-pill status-off';
    });
},1000);
document.addEventListener('click',function(){document.body.style.cursor='none';},{once:true});
</script>
</body>
</html>"""

            def _build_control_page(self):
                return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZeypherLive Control</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#e0e0e0;font-family:system-ui;min-height:100vh;padding:20px}
h1{color:#8b8bff;font-size:22px;margin-bottom:20px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{background:#1a1a2e;border:1px solid #333;border-radius:10px;padding:16px;text-align:center}
.stat-val{font-size:28px;font-weight:700;color:#55ff55}
.stat-label{color:#888;font-size:12px;margin-top:4px}
.controls{margin-top:20px;display:flex;gap:10px;flex-wrap:wrap}
.btn{padding:12px 24px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.btn-start{background:#2d6a2d;color:white}
.btn-stop{background:#6a2d2d;color:white}
</style>
</head>
<body>
<h1>ZeypherLive Control Panel</h1>
<div class="stat-grid" id="stats"></div>
<div class="controls">
<button class="btn btn-start" onclick="fetch('/status').then(r=>r.json()).then(d=>alert(JSON.stringify(d,null,2)))">Refresh Stats</button>
<button class="btn btn-stop" onclick="if(confirm('Stop stream?'))fetch('/status').then(()=>alert('Use desktop app to stop'))">Info</button>
</div>
<script>
function updateStats(){
    fetch('/status').then(r=>r.json()).then(d=>{
        document.getElementById('stats').innerHTML=`
            <div class="stat"><div class="stat-val">${d.fps}</div><div class="stat-label">FPS</div></div>
            <div class="stat"><div class="stat-val">${d.clients}</div><div class="stat-label">Viewers</div></div>
            <div class="stat"><div class="stat-val">${d.frames}</div><div class="stat-label">Total Frames</div></div>
            <div class="stat"><div class="stat-val">${Math.floor(d.uptime/60)}m</div><div class="stat-label">Uptime</div></div>
        `;
    }).catch(()=>{});
}
updateStats();
setInterval(updateStats,2000);
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
        import time as _time
        self._start_time = _time.time()
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
