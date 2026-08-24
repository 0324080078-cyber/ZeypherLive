"""ZeypherLive — Cloud Sync / WebSocket Server for Mobile & Remote"""
import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
import threading
import time
from typing import Optional, Callable
from config.settings import CONFIG


class CloudSyncServer:
    def __init__(self, config=None):
        self.config = config or CONFIG.cloud
        self._server = None
        self._clients: dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._source = None
        self._lock = threading.Lock()
        self._frame_callback: Optional[Callable] = None
        self._command_callback: Optional[Callable] = None
        self._connected_clients = 0

    def set_source(self, source):
        self._source = source

    def set_frame_callback(self, callback: Callable):
        self._frame_callback = callback

    def set_command_callback(self, callback: Callable):
        self._command_callback = callback

    async def _handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self._clients[client_id] = websocket
        self._connected_clients = len(self._clients)
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "frame":
                    frame_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None and self._frame_callback:
                        self._frame_callback(frame)
                elif msg_type == "command":
                    cmd = data.get("command", "")
                    params = data.get("params", {})
                    if self._command_callback:
                        result = self._command_callback(cmd, params)
                        await websocket.send(json.dumps({
                            "type": "command_result",
                            "command": cmd,
                            "result": result,
                        }))
                elif msg_type == "request_frame":
                    if self._source:
                        frame = self._source.read() if hasattr(self._source, 'read') else None
                        if frame is not None:
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            await websocket.send(json.dumps({
                                "type": "frame",
                                "frame": frame_b64,
                            }))
                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if client_id in self._clients:
                del self._clients[client_id]
            self._connected_clients = len(self._clients)

    async def _broadcast_frame(self, frame: np.ndarray):
        if not self._clients:
            return
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        msg = json.dumps({"type": "frame", "frame": frame_b64})
        disconnected = []
        for client_id, ws in self._clients.items():
            try:
                await ws.send(msg)
            except Exception:
                disconnected.append(client_id)
        for cid in disconnected:
            if cid in self._clients:
                del self._clients[cid]

    async def _run_server(self):
        ssl_context = None
        if self.config.use_ssl and self.config.cert_path and self.config.key_path:
            import ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.config.cert_path, self.config.key_path)
        self._server = await websockets.serve(
            self._handler,
            self.config.server_host,
            self.config.server_port,
            ssl=ssl_context,
            max_size=50 * 1024 * 1024,
        )
        await self._server.wait_closed()

    def start(self):
        if self.running:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.running = True

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_server())

    def stop(self):
        self.running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)

    def broadcast_frame(self, frame: np.ndarray):
        if self._loop and self.running:
            asyncio.run_coroutine_threadsafe(self._broadcast_frame(frame), self._loop)

    @property
    def stats(self) -> dict:
        return {
            "running": self.running,
            "connected_clients": self._connected_clients,
            "host": self.config.server_host,
            "port": self.config.server_port,
        }
