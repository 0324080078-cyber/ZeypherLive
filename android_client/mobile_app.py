"""ZeypherLive — Android Mobile Client (Kivy)"""
import os
import sys
import json
import time
import base64
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.utils import platform

Window.size = (400, 700)

try:
    import websockets
    import asyncio
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    from kivy.uix.camera import Camera
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False


class ZeypherClient:
    def __init__(self, host: str = "192.168.1.100", port: int = 8765):
        self.host = host
        self.port = port
        self.ws = None
        self.connected = False
        self._loop = None
        self._thread = None
        self._frame_callback = None
        self._command_callback = None

    def set_frame_callback(self, cb):
        self._frame_callback = cb

    def set_command_callback(self, cb):
        self._command_callback = cb

    def start(self):
        if not HAS_WS:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        try:
            uri = f"ws://{self.host}:{self.port}"
            async with websockets.connect(uri, max_size=50 * 1024 * 1024) as ws:
                self.ws = ws
                self.connected = True
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type", "")
                    if msg_type == "frame" and self._frame_callback:
                        frame_data = base64.b64decode(data["frame"])
                        self._frame_callback(frame_data)
                    elif msg_type == "command_result" and self._command_callback:
                        self._command_callback(data.get("command"), data.get("result"))
        except Exception as e:
            self.connected = False

    async def send_command(self, command: str, params: dict = None):
        if self.ws and self.connected:
            msg = json.dumps({"type": "command", "command": command, "params": params or {}})
            await self.ws.send(msg)

    def send_command_sync(self, command: str, params: dict = None):
        if self._loop:
            asyncio.run_coroutine_threadsafe(self.send_command(command, params), self._loop)

    async def request_frame(self):
        if self.ws and self.connected:
            await self.ws.send(json.dumps({"type": "request_frame"}))

    def stop(self):
        self.connected = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)


class ZeypherMobileApp(App):
    def build(self):
        self.title = "ZeypherLive Mobile"
        self.client = ZeypherClient()
        self._preview_image = None
        self._last_frame = None
        root = BoxLayout(orientation='vertical', padding=10, spacing=8)
        with root.canvas.before:
            Color(0.1, 0.1, 0.18, 1)
            self._bg_rect = Rectangle(pos=root.pos, size=root.size)
            root.bind(pos=self._update_bg, size=self._update_bg)
        header = Label(
            text="ZeypherLive Mobile",
            font_size=22,
            bold=True,
            color=(0.49, 0.49, 1.0, 1),
            size_hint_y=0.08,
        )
        root.add_widget(header)
        conn_layout = GridLayout(cols=3, size_hint_y=0.08, spacing=6)
        self.ip_input = TextInput(
            text="192.168.1.100",
            hint_text="Server IP",
            multiline=False,
            size_hint_x=0.4,
            font_size=14,
        )
        conn_layout.add_widget(self.ip_input)
        self.port_input = TextInput(
            text="8765",
            hint_text="Port",
            multiline=False,
            size_hint_x=0.2,
            font_size=14,
        )
        conn_layout.add_widget(self.port_input)
        self.connect_btn = Button(
            text="Connect",
            size_hint_x=0.3,
            font_size=14,
            bold=True,
        )
        self.connect_btn.bind(on_press=self._toggle_connection)
        conn_layout.add_widget(self.connect_btn)
        root.add_widget(conn_layout)
        self.status_label = Label(
            text="Disconnected",
            font_size=12,
            color=(1.0, 0.33, 0.33, 1),
            size_hint_y=0.04,
        )
        root.add_widget(self.status_label)
        self.preview = Image(
            size_hint_y=0.45,
            allow_stretch=True,
            keep_ratio=True,
        )
        root.add_widget(self.preview)
        controls = GridLayout(cols=2, spacing=8, size_hint_y=0.36)
        controls.add_widget(self._make_control_group("Face Swap", [
            ("Enable", "face_swap_enable"),
            ("Method", "face_swap_method"),
        ]))
        controls.add_widget(self._make_control_group("Voice", [
            ("Enable", "voice_enable"),
            ("Effect", "voice_effect"),
        ]))
        controls.add_widget(self._make_control_group("Body Track", [
            ("Enable", "body_enable"),
            ("Skeleton", "body_skeleton"),
        ]))
        controls.add_widget(self._make_control_group("OBS", [
            ("Enable", "obs_enable"),
            ("Preview", "obs_preview"),
        ]))
        root.add_widget(controls)
        presets_layout = GridLayout(cols=4, spacing=6, size_hint_y=0.06)
        for preset in ["None", "Robot", "Deep", "Alien"]:
            btn = Button(text=preset, font_size=11, bold=True)
            btn.bind(on_press=lambda inst, p=preset.lower(): self._apply_preset(p))
            presets_layout.add_widget(btn)
        root.add_widget(presets_layout)
        self.client.set_frame_callback(self._on_frame)
        Clock.schedule_interval(self._request_frame, 1.0 / 15)
        return root

    def _update_bg(self, *args):
        self._bg_rect.pos = args[0].pos if len(args) > 1 else self._bg_rect.pos
        self._bg_rect.size = args[1] if len(args) > 1 else self._bg_rect.size

    def _make_control_group(self, title, controls) -> BoxLayout:
        box = BoxLayout(orientation='vertical', spacing=4)
        with box.canvas.before:
            Color(0.09, 0.09, 0.18, 1)
            Rectangle(pos=box.pos, size=box.size)
        lbl = Label(text=title, font_size=13, bold=True, color=(0.49, 0.49, 1.0, 1), size_hint_y=0.3)
        box.add_widget(lbl)
        for ctrl_name, ctrl_id in controls:
            if "Enable" in ctrl_name:
                sw = Switch(active=False, size_hint_y=0.35)
                sw.bind(active=lambda inst, val, cid=ctrl_id: self._on_switch(cid, val))
                box.add_widget(sw)
            else:
                row = BoxLayout(orientation='horizontal', size_hint_y=0.35)
                row.add_widget(Label(text=ctrl_name + ":", font_size=11, size_hint_x=0.4))
                btn = Button(text="None", font_size=11, size_hint_x=0.6)
                btn.bind(on_press=lambda inst, cid=ctrl_id: self._cycle_option(cid))
                row.add_widget(btn)
                box.add_widget(row)
        return box

    def _toggle_connection(self, *args):
        if self.client.connected:
            self.client.stop()
            self.connect_btn.text = "Connect"
            self.status_label.text = "Disconnected"
            self.status_label.color = (1.0, 0.33, 0.33, 1)
        else:
            self.client.host = self.ip_input.text
            self.client.port = int(self.port_input.text)
            self.client.start()
            self.connect_btn.text = "Disconnect"
            self.status_label.text = "Connecting..."
            self.status_label.color = (1.0, 0.67, 0.0, 1)

    def _on_frame(self, frame_data):
        import io
        from kivy.core.image import Image as KivyImage
        from kivy.clock import Clock
        self._last_frame = frame_data
        Clock.schedule_once(lambda dt: self._update_preview())

    def _update_preview(self):
        if self._last_frame:
            import io
            from kivy.core.image import Image as KivyImage
            img = KivyImage(io.BytesIO(self._last_frame), ext='jpg')
            self.preview.texture = img.texture

    def _request_frame(self, dt):
        if self.client.connected:
            self.client.send_command_sync("request_frame")

    def _on_switch(self, ctrl_id, value):
        self.client.send_command_sync("toggle", {"module": ctrl_id, "enabled": value})

    def _cycle_option(self, ctrl_id):
        self.client.send_command_sync("cycle", {"module": ctrl_id})

    def _apply_preset(self, preset):
        presets = {
            "none": {"voice_effect": "none", "pitch": 0},
            "robot": {"voice_effect": "robot", "pitch": 0},
            "deep": {"voice_effect": "deep", "pitch": -4},
            "alien": {"voice_effect": "alien", "pitch": 3},
        }
        if preset in presets:
            self.client.send_command_sync("preset", presets[preset])

    def on_stop(self):
        self.client.stop()


if __name__ == "__main__":
    ZeypherMobileApp().run()
