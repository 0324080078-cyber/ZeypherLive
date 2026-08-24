"""ZeypherLive — Credit Billing Client"""
import time
import threading
import urllib.request
import json
from typing import Optional, Callable


class CreditBilling:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.api_key = ""
        self.token = ""
        self.credits = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._credits_per_second = 2
        self._stream_start_time = 0
        self._last_deduct_time = 0
        self._callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None

    def set_callbacks(self, on_credits: Callable = None, on_error: Callable = None):
        self._callback = on_credits
        self._error_callback = on_error

    def _notify_credits(self):
        if self._callback:
            self._callback(self.credits)

    def _notify_error(self, msg: str):
        if self._error_callback:
            self._error_callback(msg)

    def login(self, username: str, password: str) -> bool:
        try:
            data = json.dumps({"username": username, "password": password}).encode()
            req = urllib.request.Request(
                f"{self.api_url}/api/auth/login",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                self.token = result["token"]
                self.credits = result["credits"]
                self._notify_credits()
                return True
        except Exception as e:
            self._notify_error(f"Login failed: {e}")
            return False

    def register(self, username: str, email: str, password: str) -> bool:
        try:
            data = json.dumps({"username": username, "email": email, "password": password}).encode()
            req = urllib.request.Request(
                f"{self.api_url}/api/auth/register",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                self.token = result["token"]
                self.credits = result["credits"]
                self._notify_credits()
                return True
        except Exception as e:
            self._notify_error(f"Registration failed: {e}")
            return False

    def set_api_key(self, key: str):
        self.api_key = key

    def check_credits(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.api_url}/api/user/profile",
                headers={"Authorization": f"Bearer {self.token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                self.credits = result["credits"]
                self._notify_credits()
                return True
        except Exception as e:
            return False

    def can_start_stream(self) -> tuple[bool, str]:
        if not self.token:
            return False, "Not logged in"
        if self.credits < 10:
            return False, f"Need at least 10 credits. You have {self.credits}"
        return True, f"{self.credits} credits = ~{self.credits // self._credits_per_second}s"

    def start_stream_billing(self) -> bool:
        can, msg = self.can_start_stream()
        if not can:
            self._notify_error(msg)
            return False

        self._running = True
        self._stream_start_time = time.time()
        self._last_deduct_time = time.time()
        self._thread = threading.Thread(target=self._billing_loop, daemon=True)
        self._thread.start()
        return True

    def _billing_loop(self):
        while self._running:
            time.sleep(1.0)
            if not self._running:
                break

            elapsed = time.time() - self._last_deduct_time
            if elapsed >= 1.0:
                seconds = int(elapsed)
                cost = seconds * self._credits_per_second
                self._last_deduct_time = time.time()

                try:
                    data = json.dumps({"seconds": seconds}).encode()
                    req = urllib.request.Request(
                        f"{self.api_url}/api/stream/tick",
                        data=data,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.token}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        result = json.loads(resp.read())
                        self.credits = result["credits"]
                        self._notify_credits()
                except urllib.error.HTTPError as e:
                    if e.code == 402:
                        self._notify_error("Out of credits! Stream stopped.")
                        self._running = False
                        break
                except Exception:
                    pass

    def stop_stream_billing(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_billing(self) -> bool:
        return self._running

    @property
    def stream_duration(self) -> float:
        if self._stream_start_time > 0:
            return time.time() - self._stream_start_time
        return 0

    @property
    def credits_used_this_stream(self) -> int:
        return int(self.stream_duration) * self._credits_per_second
