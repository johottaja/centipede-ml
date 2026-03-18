"""Live training-progress window.

Spawns train.py as a subprocess, reads its JSON progress lines in a
background thread, and updates the UI every second via a thread-safe queue.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable


def _fmt_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


class ProgressWindow(tk.Toplevel):
    """Modal-ish window that shows live DQN training progress."""

    _POLL_MS = 300

    def __init__(self, parent: tk.Tk, cmd: list[str], on_done: Callable[[], None]):
        super().__init__(parent)
        self.title("Training in progress")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cmd = cmd
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[dict] = queue.Queue()
        self._start_time = time.monotonic()
        self._total = 1
        self._finished = False

        self._build_ui()
        self._start_process()
        self._poll()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = tk.Frame(self, padx=24, pady=20)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text="Training DQN Agent", font=("TkDefaultFont", 13, "bold")).pack(pady=(0, 16))

        self._progress_bar = ttk.Progressbar(outer, length=360, mode="determinate", maximum=100)
        self._progress_bar.pack(pady=(0, 14))

        grid = tk.Frame(outer)
        grid.pack(fill="x", pady=(0, 16))

        self._stat_vars: dict[str, tk.StringVar] = {}
        rows = [
            ("Status",         "status",    "Starting…"),
            ("Device",         "device",    "—"),
            ("Steps done",     "steps",     "—"),
            ("Steps remaining","remaining", "—"),
            ("Progress",       "pct",       "0 %"),
            ("Elapsed",        "elapsed",   "—"),
            ("ETA",            "eta",       "—"),
            ("Steps / sec",    "sps",       "—"),
        ]
        for i, (label, key, initial) in enumerate(rows):
            tk.Label(grid, text=label, anchor="w", width=18,
                     font=("TkDefaultFont", 10)).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=initial)
            self._stat_vars[key] = var
            tk.Label(grid, textvariable=var, anchor="w",
                     font=("TkDefaultFont", 10, "bold")).grid(row=i, column=1, sticky="w", padx=(8, 0), pady=2)

        self._cancel_btn = tk.Button(outer, text="Cancel Training", width=20, command=self._on_close)
        self._cancel_btn.pack(pady=(4, 0))

    # ── process management ────────────────────────────────────────────────

    def _start_process(self) -> None:
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                self._queue.put({"type": "log", "text": line})
        self._queue.put({"type": "_eof"})

    # ── polling ───────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        if not self._finished:
            self.after(self._POLL_MS, self._poll)

    def _handle_msg(self, msg: dict) -> None:
        t = msg.get("type")

        if t == "device":
            self._stat_vars["device"].set(msg["device"])

        elif t == "start":
            self._total = msg["total"]
            self._stat_vars["status"].set("Training…")
            self._stat_vars["steps"].set(f"0 / {self._total:,}")
            self._stat_vars["remaining"].set(f"{self._total:,}")

        elif t == "progress":
            steps = msg["steps"]
            total = msg["total"]
            remaining = total - steps
            self._progress_bar["value"] = msg["pct"]
            self._stat_vars["status"].set("Training…")
            self._stat_vars["steps"].set(f"{steps:,} / {total:,}")
            self._stat_vars["remaining"].set(f"{remaining:,}")
            self._stat_vars["pct"].set(f"{msg['pct']:.1f} %")
            self._stat_vars["elapsed"].set(_fmt_seconds(msg["elapsed"]))
            self._stat_vars["eta"].set(_fmt_seconds(msg["eta"]))
            self._stat_vars["sps"].set(f"{msg['steps_per_sec']:,.0f}")

        elif t == "done":
            self._progress_bar["value"] = 100
            self._stat_vars["status"].set("Done — model saved")
            self._stat_vars["pct"].set("100.0 %")
            self._stat_vars["elapsed"].set(_fmt_seconds(msg["elapsed"]))
            self._stat_vars["eta"].set("—")
            self._stat_vars["remaining"].set("0")
            self._cancel_btn.config(text="Close", command=self._close_clean)
            self._finished = True
            self._on_done()

        elif t == "_eof":
            if not self._finished:
                self._stat_vars["status"].set("Process ended")
                self._cancel_btn.config(text="Close", command=self._close_clean)
                self._finished = True
                self._on_done()

    # ── close handling ────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._close_clean()

    def _close_clean(self) -> None:
        self._finished = True
        self.destroy()
