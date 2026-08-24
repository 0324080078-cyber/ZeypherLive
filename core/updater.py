"""ZeypherLive — Auto-Updater"""
import os
import sys
import json
import shutil
import subprocess
import urllib.request
import urllib.error
import tempfile
import zipfile
import hashlib
from typing import Optional, Callable


class AutoUpdater:
    GITHUB_REPO = "0324080078-cyber/ZeypherLive"
    VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "version.json")
    CURRENT_VERSION = "1.0.0"

    def __init__(self):
        self._current_version = self._load_version()
        self._check_url = f"https://api.github.com/repos/{self.GITHUB_REPO}/releases/latest"
        self._download_url = None
        self._available_version = None
        self._changelog = ""
        self._callback: Optional[Callable] = None

    def set_callback(self, callback: Callable):
        self._callback = callback

    def _notify(self, msg: str):
        if self._callback:
            self._callback(msg)
        print(f"[Updater] {msg}")

    def _load_version(self) -> str:
        if os.path.exists(self.VERSION_FILE):
            try:
                with open(self.VERSION_FILE, "r") as f:
                    data = json.load(f)
                return data.get("version", self.CURRENT_VERSION)
            except Exception:
                pass
        return self.CURRENT_VERSION

    def _save_version(self, version: str):
        path = os.path.join(os.path.dirname(__file__), "..", "version.json")
        with open(path, "w") as f:
            json.dump({"version": version, "updated": True}, f)

    @property
    def current_version(self) -> str:
        return self._current_version

    def check_for_update(self) -> dict:
        try:
            req = urllib.request.Request(self._check_url, headers={"User-Agent": "ZeypherLive-Updater"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            tag = data.get("tag_name", "").lstrip("v")
            self._available_version = tag
            self._changelog = data.get("body", "")

            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if name.endswith(".zip") and "update" in name.lower():
                    self._download_url = asset.get("browser_download_url")
                    break

            if not self._download_url:
                for asset in data.get("assets", []):
                    name = asset.get("name", "")
                    if name.endswith(".zip"):
                        self._download_url = asset.get("browser_download_url")
                        break

            return {
                "update_available": self._compare_versions(tag, self._current_version) > 0,
                "current": self._current_version,
                "available": tag,
                "changelog": self._changelog,
                "download_url": self._download_url,
            }
        except Exception as e:
            return {
                "update_available": False,
                "current": self._current_version,
                "error": str(e),
            }

    def _compare_versions(self, v1: str, v2: str) -> int:
        def parse(v):
            return [int(x) for x in v.split(".") if x.isdigit()]
        a, b = parse(v1), parse(v2)
        for i in range(max(len(a), len(b))):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            if ai != bi:
                return 1 if ai > bi else -1
        return 0

    def download_update(self, progress_callback: Optional[Callable] = None) -> Optional[str]:
        if not self._download_url:
            self._notify("No download URL found")
            return None

        try:
            self._notify(f"Downloading update v{self._available_version}...")
            tmp_dir = tempfile.mkdtemp(prefix="zeypher_update_")
            zip_path = os.path.join(tmp_dir, "update.zip")

            def _reporthook(block, block_size, total_size):
                if progress_callback and total_size > 0:
                    pct = int(block * block_size * 100 / total_size)
                    progress_callback(pct)

            urllib.request.urlretrieve(self._download_url, zip_path, reporthook=_reporthook)
            self._notify("Download complete")

            extract_dir = os.path.join(tmp_dir, "extracted")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            return extract_dir
        except Exception as e:
            self._notify(f"Download error: {e}")
            return None

    def apply_update(self, extracted_dir: str) -> bool:
        try:
            app_dir = os.path.dirname(os.path.dirname(__file__))

            for item in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, item)
                dst = os.path.join(app_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            self._save_version(self._available_version)
            self._notify(f"Updated to v{self._available_version}")
            return True
        except Exception as e:
            self._notify(f"Update error: {e}")
            return False

    def restart_app(self):
        self._notify("Restarting...")
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def full_update(self, progress_callback: Optional[Callable] = None) -> bool:
        result = self.check_for_update()
        if not result.get("update_available"):
            self._notify("Already up to date")
            return False

        extract_dir = self.download_update(progress_callback)
        if not extract_dir:
            return False

        if self.apply_update(extract_dir):
            self.restart_app()
            return True
        return False
