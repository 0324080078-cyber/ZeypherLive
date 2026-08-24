"""ZeypherLive — Main GUI (Bulletproof)"""
import sys
import os
import cv2
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QTabWidget,
    QGroupBox, QGridLayout, QFileDialog, QStatusBar, QSpinBox,
    QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CONFIG
from core.camera import CameraCapture
from core.pipeline import FramePipeline
from body_track.tracker import BodyTracker
from face_swap.engine import FaceSwapEngine
from face_swap.local_engine import LocalFaceSwap
from voice_changer.engine import VoiceChanger
from voice_changer.realtime_engine import RealtimeVoiceChanger
from voice_changer.lip_sync import LipSyncEngine
from voice_changer.rvc_engine import RVCVoiceEngine
from obs_bridge.bridge import OBSBridge
from obs_bridge.stream_server import StreamServer
from cloud_sync.server import CloudSyncServer
from saas_backend.billing_client import CreditBilling

try:
    from lucy_engine.lucy_client import LucyEngine
    HAS_LUCY = True
except Exception as e:
    print(f"[GUI] Lucy import failed: {e}")
    HAS_LUCY = False

try:
    from lucy_engine.fal_client import FalFaceSwap
    HAS_FAL = True
except Exception as e:
    print(f"[GUI] fal.ai import failed: {e}")
    HAS_FAL = False


class WorkerSignals(QObject):
    stats_updated = pyqtSignal(dict)


DARK_STYLE = """
QMainWindow { background-color: #111111; }
QWidget { background-color: #111111; color: #e0e0e0; font-family: 'Segoe UI', Arial; }
QGroupBox { border: 1px solid #2a2a2a; border-radius: 8px; margin-top: 1em; padding-top: 12px; font-weight: bold; color: #8b8bff; font-size: 13px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton { background-color: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 8px 20px; color: #e0e0e0; font-weight: bold; font-size: 13px; }
QPushButton:hover { background-color: #0f3460; border-color: #533483; }
QPushButton:pressed { background-color: #533483; }
QPushButton:disabled { background-color: #1a1a1a; color: #555; }
QPushButton#startBtn { background-color: #2d6a2d; border-color: #3a8a3a; }
QPushButton#startBtn:hover { background-color: #3a8a3a; }
QPushButton#stopBtn { background-color: #6a2d2d; border-color: #8a3a3a; }
QPushButton#stopBtn:hover { background-color: #8a3a3a; }
QComboBox { background-color: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 6px 12px; color: #e0e0e0; font-size: 13px; min-height: 28px; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox QAbstractItemView { background-color: #1a1a2e; color: #e0e0e0; selection-background-color: #533483; border: 1px solid #333; }
QSlider::groove:horizontal { background: #333; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #7c7cff; width: 18px; height: 18px; margin: -6px 0; border-radius: 9px; }
QSlider::sub-page:horizontal { background: #533483; border-radius: 3px; }
QSpinBox, QLineEdit { background-color: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 6px; color: #e0e0e0; font-size: 13px; }
QTabWidget::pane { border: 1px solid #2a2a2a; border-radius: 6px; }
QTabBar::tab { background-color: #1a1a2e; color: #777; padding: 10px 24px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-size: 13px; font-weight: bold; }
QTabBar::tab:selected { background-color: #0f3460; color: #8b8bff; }
QStatusBar { background-color: #0a0a0a; color: #8b8bff; font-size: 12px; }
QCheckBox { spacing: 8px; font-size: 13px; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #444; border-radius: 4px; background: #1a1a2e; }
QCheckBox::indicator:checked { background-color: #533483; border-color: #8b8bff; }
QLabel { color: #e0e0e0; }
"""


class ZeypherMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeypherLive — Real-Time Face & Body Swap")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet(DARK_STYLE)
        self.signals = WorkerSignals()
        self.signals.stats_updated.connect(self._on_stats)

        self.camera = CameraCapture()
        self.pipeline = FramePipeline()
        self.body_tracker = None
        self.face_engine = FaceSwapEngine()
        self.local_face = LocalFaceSwap()
        self.voice_changer = VoiceChanger()
        self.realtime_vc = RealtimeVoiceChanger()
        self.lip_sync = LipSyncEngine()
        self.obs_bridge = OBSBridge()
        self.stream_server = StreamServer()
        self.cloud_server = CloudSyncServer()
        self.billing = CreditBilling()
        self.lucy_engine = None
        self.fal_engine = None
        self.rvc_engine = RVCVoiceEngine()

        try:
            self.body_tracker = BodyTracker()
            print("[GUI] Body tracker initialized")
        except Exception as e:
            print(f"[GUI] Body tracker failed: {e}")

        try:
            if self.local_face.initialize():
                print("[GUI] Local face swap ready")
            else:
                print("[GUI] Local face swap init failed")
        except Exception as e:
            print(f"[GUI] Local face swap error: {e}")

        if HAS_LUCY:
            self.lucy_engine = LucyEngine()
            print("[GUI] Lucy engine ready")

        self._init_ui()
        self._refresh_cameras()

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_stats)
        self._status_timer.start(1000)

        self._preview_timer = QTimer()
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(33)

        self.billing.set_callbacks(
            on_credits=lambda c: QTimer.singleShot(0, lambda: self.lbl_ai_fps.setText(f"Credits: {c}")),
            on_error=lambda e: QTimer.singleShot(0, lambda: self.statusBar().showMessage(f"Billing: {e}")),
        )

        print("[GUI] Window ready")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        top_bar = QHBoxLayout()
        title = QLabel("ZeypherLive")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #8b8bff;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        self.status_indicator = QLabel("Disconnected")
        self.status_indicator.setStyleSheet("color: #ff5555; font-size: 13px; font-weight: bold;")
        top_bar.addWidget(self.status_indicator)
        main_layout.addLayout(top_bar)

        content = QHBoxLayout()
        content.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(250)
        cam_row.addWidget(self.cam_combo, stretch=1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(80)
        btn_refresh.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(btn_refresh)
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setFixedWidth(140)
        self.btn_start.clicked.connect(self._toggle_camera)
        cam_row.addWidget(self.btn_start)
        left_col.addLayout(cam_row)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        cam_box = QVBoxLayout()
        cam_label = QLabel("YOUR CAMERA")
        cam_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        cam_label.setAlignment(Qt.AlignCenter)
        cam_box.addWidget(cam_label)
        self.preview_camera = QLabel("No camera")
        self.preview_camera.setMinimumSize(480, 360)
        self.preview_camera.setAlignment(Qt.AlignCenter)
        self.preview_camera.setStyleSheet("background-color: #0a0a0a; border: 2px solid #2a2a2a; border-radius: 8px; font-size: 14px; color: #555;")
        cam_box.addWidget(self.preview_camera, stretch=1)
        self.lbl_cam_fps = QLabel("")
        self.lbl_cam_fps.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_cam_fps.setAlignment(Qt.AlignCenter)
        cam_box.addWidget(self.lbl_cam_fps)
        preview_row.addLayout(cam_box)

        ai_box = QVBoxLayout()
        ai_label = QLabel("AI OUTPUT")
        ai_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        ai_label.setAlignment(Qt.AlignCenter)
        ai_box.addWidget(ai_label)
        self.preview_ai = QLabel("No AI output")
        self.preview_ai.setMinimumSize(480, 360)
        self.preview_ai.setAlignment(Qt.AlignCenter)
        self.preview_ai.setStyleSheet("background-color: #0a0a0a; border: 2px solid #2a2a2a; border-radius: 8px; font-size: 14px; color: #555;")
        ai_box.addWidget(self.preview_ai, stretch=1)
        self.lbl_ai_fps = QLabel("")
        self.lbl_ai_fps.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_ai_fps.setAlignment(Qt.AlignCenter)
        ai_box.addWidget(self.lbl_ai_fps)
        preview_row.addLayout(ai_box)

        left_col.addLayout(preview_row, stretch=1)
        content.addLayout(left_col, stretch=3)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        lucy_grp = QGroupBox("Lucy 2.5 Connection")
        lg = QGridLayout()
        lg.addWidget(QLabel("SaaS Server:"), 0, 0)
        self.saaS_url = QLineEdit("http://localhost:8000")
        self.saaS_url.setPlaceholderText("SaaS server URL")
        lg.addWidget(self.saaS_url, 0, 1)
        lg.addWidget(QLabel("Username:"), 1, 0)
        self.saaS_user = QLineEdit()
        self.saaS_user.setPlaceholderText("SaaS username")
        lg.addWidget(self.saaS_user, 1, 1)
        lg.addWidget(QLabel("Password:"), 2, 0)
        self.saaS_pass = QLineEdit()
        self.saaS_pass.setEchoMode(QLineEdit.Password)
        self.saaS_pass.setPlaceholderText("SaaS password")
        lg.addWidget(self.saaS_pass, 2, 1)
        self.btn_saaS_login = QPushButton("Login & Check Credits")
        self.btn_saaS_login.clicked.connect(self._saas_login)
        lg.addWidget(self.btn_saaS_login, 3, 0, 1, 2)
        self.saaS_status = QLabel("")
        self.saaS_status.setStyleSheet("color: #888; font-size: 11px;")
        lg.addWidget(self.saaS_status, 4, 0, 1, 2)
        lg.addWidget(QLabel("API Key:"), 5, 0)
        self.lucy_key = QLineEdit()
        self.lucy_key.setEchoMode(QLineEdit.Password)
        self.lucy_key.setPlaceholderText("Enter Decart API key...")
        self.lucy_key.textChanged.connect(lambda v: setattr(CONFIG.lucy, 'api_key', v))
        lg.addWidget(self.lucy_key, 5, 1)
        lg.addWidget(QLabel("Prompt:"), 6, 0)
        self.lucy_prompt = QLineEdit()
        self.lucy_prompt.setPlaceholderText("e.g. Make the person look like...")
        self.lucy_prompt.textChanged.connect(lambda v: setattr(CONFIG.lucy, 'prompt', v))
        lg.addWidget(self.lucy_prompt, 6, 1)
        ref_row = QHBoxLayout()
        self.lucy_ref_label = QLabel("No reference")
        self.lucy_ref_label.setStyleSheet("color: #888; font-size: 12px;")
        ref_row.addWidget(self.lucy_ref_label, stretch=1)
        btn_ref = QPushButton("Load Ref")
        btn_ref.setFixedWidth(90)
        btn_ref.clicked.connect(self._load_lucy_reference)
        ref_row.addWidget(btn_ref)
        lg.addLayout(ref_row, 7, 0, 1, 2)
        self.btn_lucy = QPushButton("Connect to Lucy")
        self.btn_lucy.setObjectName("startBtn")
        self.btn_lucy.clicked.connect(self._toggle_lucy)
        lg.addWidget(self.btn_lucy, 8, 0, 1, 2)
        self.lucy_status = QLabel("Disconnected")
        self.lucy_status.setStyleSheet("color: #ff5555; font-size: 12px;")
        lg.addWidget(self.lucy_status, 9, 0, 1, 2)
        lucy_grp.setLayout(lg)
        right_col.addWidget(lucy_grp)

        if HAS_FAL:
            fal_grp = QGroupBox("fal.ai Face Swap (cloud AI, cheap)")
            fg = QGridLayout()
            fg.addWidget(QLabel("API Key:"), 0, 0)
            self.fal_key = QLineEdit()
            self.fal_key.setEchoMode(QLineEdit.Password)
            self.fal_key.setPlaceholderText("fal.ai API key...")
            fg.addWidget(self.fal_key, 0, 1)
            fg.addWidget(QLabel("Model:"), 1, 0)
            self.fal_model = QComboBox()
            self.fal_model.addItems(["face-swap", "instantid", "ip-adapter", "face-couple", "face-morph"])
            fg.addWidget(self.fal_model, 1, 1)
            fg.addWidget(QLabel("Prompt:"), 2, 0)
            self.fal_prompt = QLineEdit("Substitute the character with the reference person")
            fg.addWidget(self.fal_prompt, 2, 1)
            fg.addWidget(QLabel("Strength:"), 3, 0)
            self.fal_strength = QSlider(Qt.Horizontal)
            self.fal_strength.setRange(10, 100)
            self.fal_strength.setValue(75)
            self._fal_strength_val = 0.75
            self.fal_strength.valueChanged.connect(lambda v: setattr(self, '_fal_strength_val', v / 100.0))
            fg.addWidget(self.fal_strength, 3, 1)
            self.fal_ref_label = QLabel("No reference")
            self.fal_ref_label.setStyleSheet("color: #888; font-size: 12px;")
            fg.addWidget(self.fal_ref_label, 4, 0, 1, 2)
            btn_fal_ref = QPushButton("Load Ref Image")
            btn_fal_ref.clicked.connect(self._load_fal_reference)
            fg.addWidget(btn_fal_ref, 5, 0, 1, 2)
            self.btn_fal = QPushButton("Connect fal.ai")
            self.btn_fal.setObjectName("startBtn")
            self.btn_fal.clicked.connect(self._toggle_fal)
            fg.addWidget(self.btn_fal, 6, 0, 1, 2)
            self.fal_status = QLabel("Disconnected")
            self.fal_status.setStyleSheet("color: #ff5555; font-size: 12px;")
            fg.addWidget(self.fal_status, 7, 0, 1, 2)
            fal_grp.setLayout(fg)
            right_col.addWidget(fal_grp)

        settings_tabs = QTabWidget()
        settings_tabs.addTab(self._build_face_tab(), "Face")
        settings_tabs.addTab(self._build_body_tab(), "Body")
        settings_tabs.addTab(self._build_voice_tab(), "Voice")
        settings_tabs.addTab(self._build_obs_tab(), "OBS")
        right_col.addWidget(settings_tabs, stretch=1)

        content.addLayout(right_col, stretch=2)
        main_layout.addLayout(content, stretch=1)
        self.statusBar().showMessage("Ready")

    def _build_face_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        self.face_enabled = QCheckBox("Enable Face Swap")
        self.face_enabled.setChecked(True)
        layout.addWidget(self.face_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self.face_method = QComboBox()
        methods = ["local"]
        if HAS_LUCY:
            methods.insert(0, "lucy")
        if HAS_FAL:
            methods.insert(0, "fal.ai")
        self.face_method.addItems(methods)
        row.addWidget(self.face_method)
        layout.addLayout(row)
        btn_snap = QPushButton("Snapshot as Source")
        btn_snap.clicked.connect(self._snapshot_face)
        layout.addWidget(btn_snap)
        btn_load = QPushButton("Load Face Image")
        btn_load.clicked.connect(self._load_source_face)
        layout.addWidget(btn_load)
        layout.addStretch()
        return w

    def _build_body_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        self.body_enabled = QCheckBox("Enable Body Tracking")
        self.body_enabled.setChecked(False)
        layout.addWidget(self.body_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Overlay:"))
        self.body_mode = QComboBox()
        self.body_mode.addItems(["mesh", "glow", "outline", "particles", "skeleton"])
        row.addWidget(self.body_mode)
        layout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Opacity:"))
        self.body_opacity = QSlider(Qt.Horizontal)
        self.body_opacity.setRange(10, 90)
        self.body_opacity.setValue(35)
        self.body_opacity.valueChanged.connect(self._update_body_opacity)
        row2.addWidget(self.body_opacity)
        self.opacity_lbl = QLabel("35%")
        self.opacity_lbl.setFixedWidth(40)
        self.body_opacity.valueChanged.connect(lambda v: self.opacity_lbl.setText(f"{v}%"))
        row2.addWidget(self.opacity_lbl)
        layout.addLayout(row2)
        self.body_glow = QCheckBox("Glow Effect")
        self.body_glow.setChecked(True)
        layout.addWidget(self.body_glow)
        layout.addStretch()
        return w

    def _update_body_opacity(self, value):
        if self.body_tracker:
            self.body_tracker._mesh_opacity = value / 100.0

    def _build_voice_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        grp1 = QGroupBox("Basic Voice Effects")
        g1 = QVBoxLayout(grp1)
        self.voice_enabled = QCheckBox("Enable Basic Voice Changer")
        self.voice_enabled.stateChanged.connect(self._toggle_voice)
        g1.addWidget(self.voice_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Effect:"))
        self.voice_effect = QComboBox()
        self.voice_effect.addItems(["none", "female", "male", "robot", "deep", "high", "alien", "demon", "chipmunk", "echo", "reverb", "telephone", "megaphone"])
        self.voice_effect.currentTextChanged.connect(lambda v: self.voice_changer.set_effect(v))
        row.addWidget(self.voice_effect)
        g1.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pitch:"))
        self.voice_pitch = QSlider(Qt.Horizontal)
        self.voice_pitch.setRange(-2400, 2400)
        self.voice_pitch.valueChanged.connect(lambda v: self.voice_changer.set_pitch(v / 100.0))
        row2.addWidget(self.voice_pitch)
        self.pitch_lbl = QLabel("0.0")
        self.pitch_lbl.setFixedWidth(40)
        self.voice_pitch.valueChanged.connect(lambda v: self.pitch_lbl.setText(f"{v/100:.1f}"))
        row2.addWidget(self.pitch_lbl)
        g1.addLayout(row2)
        layout.addWidget(grp1)

        grp2 = QGroupBox("Real-time Voice Changer (RVC, routes to any app)")
        g2 = QVBoxLayout(grp2)
        self.rvc_enabled = QCheckBox("Enable Real-time Voice Changer")
        self.rvc_enabled.stateChanged.connect(self._toggle_rvc)
        g2.addWidget(self.rvc_enabled)
        rvc_row = QHBoxLayout()
        rvc_row.addWidget(QLabel("Input Mic:"))
        self.rvc_input = QComboBox()
        self.rvc_input.setMinimumWidth(200)
        rvc_row.addWidget(self.rvc_input, stretch=1)
        g2.addLayout(rvc_row)
        rvc_row2 = QHBoxLayout()
        rvc_row2.addWidget(QLabel("Pitch:"))
        self.rvc_pitch = QSlider(Qt.Horizontal)
        self.rvc_pitch.setRange(-2400, 2400)
        self.rvc_pitch.setValue(400)
        self.rvc_pitch.valueChanged.connect(lambda v: setattr(self.realtime_vc, '_pitch_shift', v / 100.0))
        rvc_row2.addWidget(self.rvc_pitch)
        self.rvc_pitch_lbl = QLabel("4.0")
        self.rvc_pitch_lbl.setFixedWidth(40)
        self.rvc_pitch.valueChanged.connect(lambda v: self.rvc_pitch_lbl.setText(f"{v/100:.1f}"))
        rvc_row2.addWidget(self.rvc_pitch_lbl)
        g2.addLayout(rvc_row2)
        rvc_row3 = QHBoxLayout()
        rvc_row3.addWidget(QLabel("Formant:"))
        self.rvc_formant = QSlider(Qt.Horizontal)
        self.rvc_formant.setRange(-100, 100)
        self.rvc_formant.setValue(30)
        self.rvc_formant.valueChanged.connect(lambda v: setattr(self.realtime_vc, '_formant_shift', v / 100.0))
        rvc_row3.addWidget(self.rvc_formant)
        self.rvc_formant_lbl = QLabel("0.3")
        self.rvc_formant_lbl.setFixedWidth(40)
        self.rvc_formant.valueChanged.connect(lambda v: self.rvc_formant_lbl.setText(f"{v/100:.1f}"))
        rvc_row3.addWidget(self.rvc_formant_lbl)
        g2.addLayout(rvc_row3)
        self.btn_rvc = QPushButton("Start Real-time Voice Changer")
        self.btn_rvc.setObjectName("startBtn")
        self.btn_rvc.clicked.connect(self._toggle_rvc_start)
        g2.addWidget(self.btn_rvc)
        self.rvc_status = QLabel("Select input device and start")
        self.rvc_status.setStyleSheet("color: #888; font-size: 11px;")
        g2.addWidget(self.rvc_status)
        layout.addWidget(grp2)

        grp3 = QGroupBox("Lip Sync (Audio drives mouth movement)")
        g3 = QVBoxLayout(grp3)
        self.lip_sync_enabled = QCheckBox("Enable Lip Sync")
        self.lip_sync_enabled.stateChanged.connect(self._toggle_lip_sync)
        g3.addWidget(self.lip_sync_enabled)
        lip_row = QHBoxLayout()
        lip_row.addWidget(QLabel("Mic:"))
        self.lip_sync_input = QComboBox()
        self.lip_sync_input.setMinimumWidth(200)
        lip_row.addWidget(self.lip_sync_input, stretch=1)
        g3.addLayout(lip_row)
        btn_lip_start = QPushButton("Start Lip Sync")
        btn_lip_start.setObjectName("startBtn")
        btn_lip_start.clicked.connect(self._toggle_lip_sync_start)
        self.btn_lip_start = btn_lip_start
        g3.addWidget(btn_lip_start)
        self.lip_sync_status = QLabel("Select mic and start")
        self.lip_sync_status.setStyleSheet("color: #888; font-size: 11px;")
        g3.addWidget(self.lip_sync_status)
        layout.addWidget(grp3)

        grp4 = QGroupBox("RVC Voice Cloning (free local, needs GPU)")
        g4 = QVBoxLayout(grp4)
        self.rvc_model_path = QLineEdit()
        self.rvc_model_path.setPlaceholderText("Path to RVC .pth model file...")
        g4.addWidget(self.rvc_model_path)
        btn_rvc_model = QPushButton("Load Model")
        btn_rvc_model.clicked.connect(self._load_rvc_model)
        g4.addWidget(btn_rvc_model)
        rvc_row4 = QHBoxLayout()
        rvc_row4.addWidget(QLabel("Pitch Shift:"))
        self.rvc_pitch_shift = QSlider(Qt.Horizontal)
        self.rvc_pitch_shift.setRange(-12, 12)
        self.rvc_pitch_shift.setValue(0)
        self.rvc_pitch_shift.valueChanged.connect(lambda v: setattr(self.rvc_engine, '_pitch_shift', v))
        rvc_row4.addWidget(self.rvc_pitch_shift)
        self.rvc_pitch_lbl2 = QLabel("0")
        self.rvc_pitch_lbl2.setFixedWidth(30)
        self.rvc_pitch_shift.valueChanged.connect(lambda v: self.rvc_pitch_lbl2.setText(str(v)))
        rvc_row4.addWidget(self.rvc_pitch_lbl2)
        g4.addLayout(rvc_row4)
        self.rvc_engine_status = QLabel("No model loaded")
        self.rvc_engine_status.setStyleSheet("color: #888; font-size: 11px;")
        g4.addWidget(self.rvc_engine_status)
        layout.addWidget(grp4)

        layout.addStretch()
        return w

    def _build_obs_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        grp1 = QGroupBox("OBS Virtual Camera (for desktop video calls)")
        g1 = QVBoxLayout(grp1)
        self.obs_enabled = QCheckBox("Enable OBS Bridge")
        self.obs_enabled.setChecked(True)
        g1.addWidget(self.obs_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Resolution:"))
        self.obs_w = QSpinBox()
        self.obs_w.setRange(320, 3840)
        self.obs_w.setValue(1280)
        row.addWidget(self.obs_w)
        row.addWidget(QLabel("x"))
        self.obs_h = QSpinBox()
        self.obs_h.setRange(240, 2160)
        self.obs_h.setValue(720)
        row.addWidget(self.obs_h)
        g1.addLayout(row)
        self.btn_obs = QPushButton("Start OBS Bridge")
        self.btn_obs.setObjectName("startBtn")
        self.btn_obs.clicked.connect(self._toggle_obs)
        g1.addWidget(self.btn_obs)
        self.obs_status = QLabel("Requires OBS Studio installed")
        self.obs_status.setStyleSheet("color: #888; font-size: 11px;")
        g1.addWidget(self.obs_status)
        layout.addWidget(grp1)

        grp2 = QGroupBox("Phone / Browser Viewer (any device on WiFi)")
        g2 = QVBoxLayout(grp2)
        self.btn_stream = QPushButton("Start Stream Server")
        self.btn_stream.setObjectName("startBtn")
        self.btn_stream.clicked.connect(self._toggle_stream)
        g2.addWidget(self.btn_stream)
        self.stream_url = QLabel("")
        self.stream_url.setStyleSheet("color: #55ff55; font-size: 12px; font-weight: bold;")
        self.stream_url.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g2.addWidget(self.stream_url)
        self.stream_status = QLabel("Open this URL on your phone browser")
        self.stream_status.setStyleSheet("color: #888; font-size: 11px;")
        g2.addWidget(self.stream_status)
        layout.addWidget(grp2)
        layout.addStretch()
        return w

    def _refresh_cameras(self):
        self.cam_combo.clear()
        try:
            devices = self.camera.list_devices()
            for d in devices:
                self.cam_combo.addItem(f"Camera {d['id']} ({d['width']}x{d['height']})", d['id'])
            if not devices:
                self.cam_combo.addItem("No camera found", -1)
        except Exception as e:
            print(f"[GUI] Camera refresh failed: {e}")
            self.cam_combo.addItem("Error detecting cameras", -1)

    def _toggle_camera(self):
        try:
            if self.camera.running:
                self.camera.stop()
                self.pipeline.stop()
                self.btn_start.setText("Start Camera")
                self.btn_start.setObjectName("startBtn")
                self.btn_start.style().polish(self.btn_start)
                self.statusBar().showMessage("Camera stopped")
                print("[GUI] Camera stopped")
            else:
                idx = self.cam_combo.currentData()
                if idx is None or idx < 0:
                    QMessageBox.warning(self, "Camera", "No camera selected.")
                    return
                print(f"[GUI] Opening camera {idx}...")
                CONFIG.camera.device_id = idx
                if not self.camera.open(idx):
                    reply = QMessageBox.question(
                        self, "Camera",
                        f"Camera {idx} failed.\nTry next working camera?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self._try_next_camera(idx)
                    return
                time.sleep(0.3)
                print(f"[GUI] Camera opened. Starting capture...")
                self.camera.start()
                time.sleep(0.5)
                frame = self.camera.read()
                if frame is None:
                    QMessageBox.warning(self, "Camera", f"Camera {idx} opened but can't read frames.\nIt may be in use by another app.")
                    self.camera.stop()
                    return
                print(f"[GUI] Capture started. FPS: {self.camera.fps_actual}")
                self.btn_start.setText("Stop Camera")
                self.btn_start.setObjectName("stopBtn")
                self.btn_start.style().polish(self.btn_start)
                self.statusBar().showMessage(f"Camera {idx} running")
                print(f"[GUI] Camera {idx} running OK")
        except Exception as e:
            print(f"[GUI] Camera toggle error: {e}")
            traceback.print_exc()

    def _try_next_camera(self, failed_idx):
        devices = self.camera.list_devices()
        for d in devices:
            if d["id"] != failed_idx:
                print(f"[GUI] Trying camera {d['id']}...")
                CONFIG.camera.device_id = d["id"]
                if self.camera.open(d["id"]):
                    time.sleep(0.3)
                    self.camera.start()
                    time.sleep(0.5)
                    frame = self.camera.read()
                    if frame is not None:
                        self.cam_combo.setCurrentIndex(
                            next((i for i in range(self.cam_combo.count())
                                  if self.cam_combo.itemData(i) == d["id"]), 0)
                        )
                        self.btn_start.setText("Stop Camera")
                        self.btn_start.setObjectName("stopBtn")
                        self.btn_start.style().polish(self.btn_start)
                        self.statusBar().showMessage(f"Camera {d['id']} running")
                        print(f"[GUI] Camera {d['id']} working")
                        return
                    self.camera.stop()
        QMessageBox.warning(self, "Camera", "No working camera found.\nClose other apps using the camera.")

    def _update_preview(self):
        try:
            cam_frame = self.camera.read()
            if cam_frame is not None:
                display_frame = cam_frame.copy()

                if self.body_tracker is not None and self.body_enabled.isChecked():
                    try:
                        result = self.body_tracker.process(display_frame)
                        if result is not None:
                            mode = self.body_mode.currentText()
                            if mode == "skeleton":
                                display_frame = self.body_tracker.draw_landmarks(display_frame, result)
                            else:
                                if self.body_glow.isChecked():
                                    self.body_tracker._glow_enabled = True
                                else:
                                    self.body_tracker._glow_enabled = False
                                display_frame = self.body_tracker.apply_body_overlay(display_frame, result, mode=mode)
                    except Exception as e:
                        pass

                if self.face_enabled.isChecked():
                    method = self.face_method.currentText()
                    if method == "local" and self.local_face.available and self.local_face.has_source:
                        try:
                            swapped = self.local_face.process(display_frame)
                            if swapped is not None:
                                display_frame = swapped
                        except Exception as e:
                            pass

                if self.lip_sync.running:
                    try:
                        landmarks = self.local_face.get_face_landmarks(display_frame)
                        if landmarks is not None:
                            display_frame = self.lip_sync.apply_lip_sync_to_frame(display_frame, landmarks)
                    except Exception:
                        pass

                self._display_frame(display_frame, self.preview_camera)
                self.lbl_cam_fps.setText(f"FPS: {self.camera.fps_actual:.0f}")

                if self.lucy_engine and self.lucy_engine.connected:
                    self.lucy_engine.push_frame(cam_frame)
            elif self.camera.running:
                self.lbl_cam_fps.setText("Waiting for frames...")
            else:
                self.lbl_cam_fps.setText("")
        except Exception as e:
            self.lbl_cam_fps.setText(f"Error: {e}")

        try:
            ai_frame = None
            if self.lucy_engine and self.lucy_engine.connected:
                ai_frame = self.lucy_engine.read()
                if ai_frame is not None:
                    self._display_frame(ai_frame, self.preview_ai)
                    self.lbl_ai_fps.setText(f"Lucy — Sent: {self.lucy_engine._frames_sent} | RX: {self.lucy_engine._frames_received}")
                else:
                    self.lbl_ai_fps.setText("Waiting for Lucy frames...")
            elif self.fal_engine and self.fal_engine.connected:
                self.fal_engine.push_frame(cam_frame)
                ai_frame = self.fal_engine.read()
                if ai_frame is not None:
                    self._display_frame(ai_frame, self.preview_ai)
                    stats = self.fal_engine.stats
                    self.lbl_ai_fps.setText(f"fal.ai — {stats['model_name']} — sent:{stats['frames_sent']} rx:{stats['frames_received']}")
                else:
                    busy = "processing..." if self.fal_engine._busy else "ready"
                    err = f" [{self.fal_engine._last_error}]" if self.fal_engine._last_error else ""
                    self.lbl_ai_fps.setText(f"fal.ai — {busy}{err}")
            elif cam_frame is not None:
                ai_frame = cam_frame
                self._display_frame(cam_frame, self.preview_ai)
                self.lbl_ai_fps.setText("Connect Lucy or fal.ai for AI swap")
            else:
                self.preview_ai.setText("No output")
                self.lbl_ai_fps.setText("")

            if self.stream_server.running and ai_frame is not None:
                self.stream_server.push_frame(ai_frame)
        except Exception as e:
            self.lbl_ai_fps.setText(f"Error: {e}")

    def _display_frame(self, frame, label):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            data = rgb.tobytes()
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
        except Exception as e:
            label.setText(f"Display error: {e}")

    def _snapshot_face(self):
        frame = self.camera.read()
        if frame is None:
            QMessageBox.warning(self, "Face", "No camera frame. Start camera first.")
            return
        try:
            if self.face_engine.set_source_from_frame(frame):
                self.statusBar().showMessage("Source face set from snapshot")
            else:
                QMessageBox.warning(self, "Face", "No face detected. Make sure your face is visible.")
        except Exception as e:
            QMessageBox.warning(self, "Face Error", str(e))

    def _load_source_face(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Face", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            try:
                if self.local_face.set_source_face(path):
                    self.statusBar().showMessage(f"Face loaded: {os.path.basename(path)}")
                else:
                    QMessageBox.warning(self, "Face", "Could not load face.\n\nTips:\n- Use a clear, front-facing photo\n- Make sure the face is well-lit\n- Try a different image")
            except Exception as e:
                QMessageBox.warning(self, "Face Error", str(e))

    def _load_lucy_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            CONFIG.lucy.reference_image = path
            self.lucy_ref_label.setText(os.path.basename(path))
            self.lucy_ref_label.setStyleSheet("color: #8b8bff; font-size: 12px;")

    def _saas_login(self):
        url = self.saaS_url.text().strip()
        user = self.saaS_user.text().strip()
        pw = self.saaS_pass.text().strip()
        if not url or not user or not pw:
            self.saaS_status.setText("Fill in all fields")
            self.saaS_status.setStyleSheet("color: #ff5555; font-size: 11px;")
            return
        self.billing.api_url = url
        self.saaS_status.setText("Logging in...")
        self.saaS_status.setStyleSheet("color: #ffaa00; font-size: 11px;")
        import threading
        def _do_login():
            ok = self.billing.login(user, pw)
            if ok:
                QTimer.singleShot(0, lambda: self.saaS_status.setText(f"Logged in — {self.billing.credits} credits"))
                QTimer.singleShot(0, lambda: self.saaS_status.setStyleSheet("color: #55ff55; font-size: 11px;"))
            else:
                QTimer.singleShot(0, lambda: self.saaS_status.setText("Login failed"))
                QTimer.singleShot(0, lambda: self.saaS_status.setStyleSheet("color: #ff5555; font-size: 11px;"))
        threading.Thread(target=_do_login, daemon=True).start()

    def _toggle_voice(self, state):
        try:
            if state == Qt.Checked:
                if not self.voice_changer._pa:
                    self.voice_changer.initialize()
                self.voice_changer.start()
            else:
                self.voice_changer.stop()
        except Exception as e:
            print(f"[GUI] Voice error: {e}")

    def _toggle_rvc(self, state):
        if state == Qt.Checked:
            devices = self.realtime_vc.list_devices()
            self.rvc_input.clear()
            for d in devices:
                self.rvc_input.addItem(f"{d['name']} ({d['rate']}Hz)", d['id'])
        else:
            self.rvc_input.clear()

    def _toggle_rvc_start(self):
        if self.realtime_vc.running:
            self.realtime_vc.stop()
            self.btn_rvc.setText("Start Real-time Voice Changer")
            self.btn_rvc.setObjectName("startBtn")
            self.btn_rvc.style().polish(self.btn_rvc)
            self.rvc_status.setText("Stopped")
            self.rvc_status.setStyleSheet("color: #ff5555; font-size: 11px;")
        else:
            dev_id = self.rvc_input.currentData()
            if dev_id is None:
                self.rvc_status.setText("Select an input device first")
                self.rvc_status.setStyleSheet("color: #ff5555; font-size: 11px;")
                return
            self.realtime_vc.set_devices(input_id=dev_id)
            if self.realtime_vc.start():
                self.btn_rvc.setText("Stop Real-time Voice Changer")
                self.btn_rvc.setObjectName("stopBtn")
                self.btn_rvc.style().polish(self.btn_rvc)
                self.rvc_status.setText("Running — route app mic to this output")
                self.rvc_status.setStyleSheet("color: #55ff55; font-size: 11px;")
            else:
                self.rvc_status.setText("Failed to start — check device")
                self.rvc_status.setStyleSheet("color: #ff5555; font-size: 11px;")

    def _toggle_lip_sync(self, state):
        if state == Qt.Checked:
            devices = self.lip_sync.list_devices()
            self.lip_sync_input.clear()
            for d in devices:
                self.lip_sync_input.addItem(f"{d['name']} ({d['rate']}Hz)", d['id'])
        else:
            self.lip_sync_input.clear()

    def _toggle_lip_sync_start(self):
        if self.lip_sync.running:
            self.lip_sync.stop()
            self.btn_lip_start.setText("Start Lip Sync")
            self.btn_lip_start.setObjectName("startBtn")
            self.btn_lip_start.style().polish(self.btn_lip_start)
            self.lip_sync_status.setText("Stopped")
            self.lip_sync_status.setStyleSheet("color: #ff5555; font-size: 11px;")
        else:
            dev_id = self.lip_sync_input.currentData()
            if dev_id is None:
                self.lip_sync_status.setText("Select a mic first")
                self.lip_sync_status.setStyleSheet("color: #ff5555; font-size: 11px;")
                return
            self.lip_sync.set_device(dev_id)
            if self.lip_sync.start():
                self.btn_lip_start.setText("Stop Lip Sync")
                self.btn_lip_start.setObjectName("stopBtn")
                self.btn_lip_start.style().polish(self.btn_lip_start)
                self.lip_sync_status.setText("Running — mouth tracks your voice")
                self.lip_sync_status.setStyleSheet("color: #55ff55; font-size: 11px;")
            else:
                self.lip_sync_status.setText("Failed to start")
                self.lip_sync_status.setStyleSheet("color: #ff5555; font-size: 11px;")

    def _toggle_lucy(self):
        if not HAS_LUCY:
            QMessageBox.warning(self, "Lucy", "Lucy module not available.")
            return
        try:
            if self.lucy_engine and self.lucy_engine.connected:
                self.lucy_engine.stop()
                self.billing.stop_stream_billing()
                self.lucy_status.setText("Disconnected")
                self.lucy_status.setStyleSheet("color: #ff5555; font-size: 12px;")
                self.btn_lucy.setText("Connect to Lucy")
                self.status_indicator.setText("Disconnected")
                self.status_indicator.setStyleSheet("color: #ff5555; font-size: 13px; font-weight: bold;")
            else:
                if not CONFIG.lucy.api_key:
                    QMessageBox.warning(self, "Lucy", "Enter your Decart API key first.")
                    return
                if self.billing.token:
                    can, msg = self.billing.can_start_stream()
                    if not can:
                        QMessageBox.warning(self, "Credits", msg)
                        return
                    self.billing.start_stream_billing()
                self.lucy_status.setText("Connecting...")
                self.lucy_status.setStyleSheet("color: #ffaa00; font-size: 12px;")
                if self.lucy_engine is None:
                    self.lucy_engine = LucyEngine()
                self.lucy_engine.start()
                self._lucy_check_attempts = 0
                QTimer.singleShot(2000, self._check_lucy)
        except Exception as e:
            print(f"[GUI] Lucy error: {e}")
            traceback.print_exc()

    def _check_lucy(self):
        if self.lucy_engine and self.lucy_engine.connected:
            self.lucy_status.setText("Connected")
            self.lucy_status.setStyleSheet("color: #55ff55; font-size: 12px;")
            self.btn_lucy.setText("Disconnect")
            self.status_indicator.setText("Live")
            self.status_indicator.setStyleSheet("color: #55ff55; font-size: 13px; font-weight: bold;")
        else:
            self._lucy_check_attempts = getattr(self, '_lucy_check_attempts', 0) + 1
            if self._lucy_check_attempts < 5:
                QTimer.singleShot(2000, self._check_lucy)
            else:
                err = self.lucy_engine._last_error if self.lucy_engine else "Unknown error"
                self.lucy_status.setText(f"Failed: {err}")
                self.lucy_status.setStyleSheet("color: #ff5555; font-size: 12px;")

    def _toggle_obs(self):
        try:
            if self.obs_bridge.running:
                self.obs_bridge.stop()
                self.btn_obs.setText("Start OBS Bridge")
                self.btn_obs.setObjectName("startBtn")
                self.btn_obs.style().polish(self.btn_obs)
                self.obs_status.setText("Stopped")
                self.obs_status.setStyleSheet("color: #ff5555; font-size: 12px;")
            else:
                self.obs_bridge.set_source(self.camera)
                CONFIG.obs.width = self.obs_w.value()
                CONFIG.obs.height = self.obs_h.value()
                if self.obs_bridge.start():
                    self.btn_obs.setText("Stop OBS Bridge")
                    self.btn_obs.setObjectName("stopBtn")
                    self.btn_obs.style().polish(self.btn_obs)
                    self.obs_status.setText("Running")
                    self.obs_status.setStyleSheet("color: #55ff55; font-size: 12px;")
                else:
                    self.obs_status.setText("Failed — start OBS first")
                    self.obs_status.setStyleSheet("color: #ff5555; font-size: 12px;")
        except Exception as e:
            print(f"[GUI] OBS error: {e}")

    def _toggle_stream(self):
        try:
            if self.stream_server.running:
                self.stream_server.stop()
                self.btn_stream.setText("Start Stream Server")
                self.btn_stream.setObjectName("startBtn")
                self.btn_stream.style().polish(self.btn_stream)
                self.stream_url.setText("")
                self.stream_status.setText("Open this URL on your phone browser")
                self.stream_status.setStyleSheet("color: #888; font-size: 11px;")
            else:
                if self.stream_server.start():
                    self.btn_stream.setText("Stop Stream Server")
                    self.btn_stream.setObjectName("stopBtn")
                    self.btn_stream.style().polish(self.btn_stream)
                    url = self.stream_server.url
                    self.stream_url.setText(url)
                    self.stream_status.setText("Open this URL on your phone browser")
                    self.stream_status.setStyleSheet("color: #55ff55; font-size: 11px;")
                else:
                    self.stream_status.setText("Failed to start")
                    self.stream_status.setStyleSheet("color: #ff5555; font-size: 11px;")
        except Exception as e:
            print(f"[GUI] Stream error: {e}")

    def _load_fal_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load fal.ai Reference", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            if self.fal_engine is None:
                self.fal_engine = FalFaceSwap()
            self.fal_engine.set_reference(path)
            self.fal_ref_label.setText(os.path.basename(path))
            self.fal_ref_label.setStyleSheet("color: #55ff55; font-size: 12px;")

    def _toggle_fal(self):
        if not HAS_FAL:
            QMessageBox.warning(self, "fal.ai", "fal.ai module not available.")
            return
        try:
            if self.fal_engine and self.fal_engine.connected:
                self.fal_engine.disconnect()
                self.fal_status.setText("Disconnected")
                self.fal_status.setStyleSheet("color: #ff5555; font-size: 12px;")
                self.btn_fal.setText("Connect fal.ai")
            else:
                key = self.fal_key.text().strip()
                if not key:
                    QMessageBox.warning(self, "fal.ai", "Enter your fal.ai API key first.")
                    return
                model = self.fal_model.currentText()
                if self.fal_engine is None:
                    self.fal_engine = FalFaceSwap()
                self.fal_engine.set_model(model)
                self.fal_engine._prompt = self.fal_prompt.text()
                self.fal_engine._strength = self._fal_strength_val
                self.fal_status.setText("Connecting...")
                self.fal_status.setStyleSheet("color: #ffaa00; font-size: 12px;")
                if self.fal_engine.connect(key):
                    self.fal_status.setText(f"Connected — {self.fal_engine.MODELS[model]['name']}")
                    self.fal_status.setStyleSheet("color: #55ff55; font-size: 12px;")
                    self.btn_fal.setText("Disconnect fal.ai")
                else:
                    err = self.fal_engine.last_error
                    self.fal_status.setText(f"Failed: {err}")
                    self.fal_status.setStyleSheet("color: #ff5555; font-size: 12px;")
        except Exception as e:
            print(f"[GUI] fal.ai error: {e}")
            traceback.print_exc()

    def _load_rvc_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RVC Model", "", "Models (*.pth)")
        if path:
            if self.rvc_engine.load_model(path):
                self.rvc_engine_status.setText(f"Loaded: {os.path.basename(path)}")
                self.rvc_engine_status.setStyleSheet("color: #55ff55; font-size: 11px;")
            else:
                self.rvc_engine_status.setText("Load failed — install torch")
                self.rvc_engine_status.setStyleSheet("color: #ff5555; font-size: 11px;")

    def _update_stats(self):
        pass

    def _on_stats(self, stats):
        pass

    def closeEvent(self, event):
        try:
            self.camera.stop()
            self.pipeline.stop()
            self.voice_changer.stop()
            self.realtime_vc.stop()
            self.lip_sync.stop()
            self.obs_bridge.stop()
            self.stream_server.stop()
            self.cloud_server.stop()
            self.billing.stop_stream_billing()
            if self.lucy_engine:
                self.lucy_engine.stop()
            if self.fal_engine:
                self.fal_engine.disconnect()
            if self.rvc_engine.running:
                self.rvc_engine.stop()
            CONFIG.save()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ZeypherLive")
    window = ZeypherMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
