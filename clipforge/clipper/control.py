"""Pause / resume / cancel for a running job.

Calling the object (it is passed around as the `cancelled` callback) blocks while the job is paused and
returns True once the job is cancelled. A running ffmpeg process is frozen/unfrozen with psutil, so pausing
takes effect immediately even in the middle of a render."""
from __future__ import annotations

import threading
import time


class Control:
    def __init__(self):
        self.cancelled = False
        self.paused = False
        self._proc = None
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        while self.paused and not self.cancelled:
            time.sleep(0.25)
        return self.cancelled

    def attach(self, proc):
        with self._lock:
            self._proc = proc
            if self.paused:
                self._suspend(True)

    def detach(self):
        with self._lock:
            self._proc = None

    def pause(self):
        with self._lock:
            self.paused = True
            self._suspend(True)

    def resume(self):
        with self._lock:
            self.paused = False
            self._suspend(False)

    def cancel(self):
        with self._lock:
            self.cancelled = True
            self.paused = False
            self._suspend(False)
            if self._proc is not None:        # don't wait for ffmpeg's next output line
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def _suspend(self, on: bool):
        if self._proc is None:
            return
        try:
            import psutil
            p = psutil.Process(self._proc.pid)
            p.suspend() if on else p.resume()
        except Exception:
            pass  # without psutil, pausing still works between steps and when ffmpeg's output buffer fills
