"""Live training-progress window.

Spawns train.py as a subprocess, reads its JSON progress lines in a
background thread, and updates the UI every second via a thread-safe queue.
"""
from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
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


def _fmt_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M"
    if steps >= 1_000:
        return f"{steps / 1_000:.0f}k"
    return str(steps)


class EvalScoreChart(tk.Canvas):
    """Simple line chart for eval mean scores over training steps."""

    _MARGIN_LEFT = 56
    _MARGIN_RIGHT = 12
    _MARGIN_TOP = 16
    _MARGIN_BOTTOM = 28
    _Y_STEP = 10_000
    _X_STEP = 1_000_000

    def __init__(self, parent: tk.Widget, width: int = 420, height: int = 180,
                 eval_freq: int = 30_000, n_eval_episodes: int = 10, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
            **kwargs,
        )
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._total_steps = 1_000_000
        self._points: list[tuple[int, float]] = []
        self.bind("<Configure>", self._on_resize)

    def set_total_steps(self, total: int) -> None:
        self._total_steps = max(total, 1)
        self._redraw()

    def add_point(self, steps: int, mean_score: float) -> None:
        self._points.append((steps, mean_score))
        self._redraw()

    def _on_resize(self, _event: tk.Event) -> None:
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)

        plot_l = self._MARGIN_LEFT
        plot_r = w - self._MARGIN_RIGHT
        plot_t = self._MARGIN_TOP
        plot_b = h - self._MARGIN_BOTTOM
        plot_w = max(plot_r - plot_l, 1)
        plot_h = max(plot_b - plot_t, 1)

        self.create_text(
            w // 2, 8,
            text=f"Game score ({self._n_eval_episodes}-game avg, deterministic)",
            font=("TkDefaultFont", 9, "bold"),
            fill="#333333",
        )

        if not self._points:
            steps_max = max(
                math.ceil(self._total_steps / self._X_STEP) * self._X_STEP,
                self._X_STEP,
            )
            steps_min = 0
            y_min = 0
            y_max = self._Y_STEP

            def x_pos_empty(steps: int) -> float:
                return plot_l + (steps - steps_min) / (steps_max - steps_min) * plot_w

            def y_pos_empty(score: float) -> float:
                return plot_b - (score - y_min) / (y_max - y_min) * plot_h

            self.create_line(plot_l, plot_t, plot_l, plot_b, fill="#999999")
            self.create_line(plot_l, plot_b, plot_r, plot_b, fill="#999999")

            y = y_min
            while y <= y_max + 0.001:
                yp = y_pos_empty(y)
                self.create_line(plot_l, yp, plot_r, yp, fill="#e8e8e8")
                self.create_line(plot_l - 3, yp, plot_l, yp, fill="#999999")
                label = "0" if y == 0 else (f"{y / 1_000:.0f}k" if y >= 1_000 else str(int(y)))
                self.create_text(
                    plot_l - 6, yp, text=label, anchor="e",
                    font=("TkDefaultFont", 8), fill="#666666",
                )
                y += self._Y_STEP

            steps = steps_min
            while steps <= steps_max:
                xp = x_pos_empty(steps)
                self.create_line(xp, plot_t, xp, plot_b, fill="#e8e8e8")
                self.create_line(xp, plot_b, xp, plot_b + 3, fill="#999999")
                self.create_text(
                    xp, plot_b + 10, text=_fmt_steps(steps),
                    font=("TkDefaultFont", 8), fill="#666666",
                )
                steps += self._X_STEP

            self.create_text(
                w // 2, (plot_t + plot_b) // 2,
                text=f"Waiting for first evaluation at {_fmt_steps(self._eval_freq)} steps…",
                font=("TkDefaultFont", 9),
                fill="#888888",
            )
            return

        scores = [p[1] for p in self._points]
        y_min = math.floor(min(scores) / self._Y_STEP) * self._Y_STEP
        y_max = math.ceil(max(scores) / self._Y_STEP) * self._Y_STEP
        if y_max <= y_min:
            y_max = y_min + self._Y_STEP

        steps_max = max(self._total_steps, self._points[-1][0])
        steps_max = max(
            math.ceil(steps_max / self._X_STEP) * self._X_STEP,
            self._X_STEP,
        )
        steps_min = 0

        def x_pos(steps: int) -> float:
            return plot_l + (steps - steps_min) / (steps_max - steps_min) * plot_w

        def y_pos(score: float) -> float:
            return plot_b - (score - y_min) / (y_max - y_min) * plot_h

        # Axes
        self.create_line(plot_l, plot_t, plot_l, plot_b, fill="#999999")
        self.create_line(plot_l, plot_b, plot_r, plot_b, fill="#999999")

        # Y-axis grid and labels every 10k score
        y = y_min
        while y <= y_max + 0.001:
            yp = y_pos(y)
            self.create_line(plot_l, yp, plot_r, yp, fill="#e8e8e8")
            self.create_line(plot_l - 3, yp, plot_l, yp, fill="#999999")
            label = "0" if y == 0 else (f"{y / 1_000:.0f}k" if y >= 1_000 else str(int(y)))
            self.create_text(
                plot_l - 6, yp, text=label, anchor="e",
                font=("TkDefaultFont", 8), fill="#666666",
            )
            y += self._Y_STEP

        # X-axis grid and labels every 1M steps
        steps = steps_min
        while steps <= steps_max:
            xp = x_pos(steps)
            self.create_line(xp, plot_t, xp, plot_b, fill="#e8e8e8")
            self.create_line(xp, plot_b, xp, plot_b + 3, fill="#999999")
            self.create_text(
                xp, plot_b + 10, text=_fmt_steps(steps),
                font=("TkDefaultFont", 8), fill="#666666",
            )
            steps += self._X_STEP

        # Data line and points
        coords: list[float] = []
        for steps, score in self._points:
            coords.extend((x_pos(steps), y_pos(score)))
        if len(coords) >= 4:
            self.create_line(*coords, fill="#2563eb", width=2, smooth=False)
        for steps, score in self._points:
            x, y = x_pos(steps), y_pos(score)
            self.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#2563eb", outline="#1d4ed8")


class ProgressWindow(tk.Toplevel):
    """Modal-ish window that shows live DQN training progress."""

    _POLL_MS = 300

    def __init__(self, parent: tk.Tk, cmd: list[str], on_done: Callable[[], None],
                 eval_freq: int = 30_000, n_eval_episodes: int = 10):
        super().__init__(parent)
        self.title("Training in progress")
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self.resizable(True, True)
        self.minsize(420, 520)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cmd = cmd
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[dict] = queue.Queue()
        self._start_time = time.monotonic()
        self._total = 1
        self._finished = False
        self._done_cb_called = False

        self._build_ui()
        self._start_process()
        self._poll()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = tk.Frame(self, padx=24, pady=20)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text="Training DQN Agent", font=("TkDefaultFont", 13, "bold")).pack(pady=(0, 16))

        self._progress_bar = ttk.Progressbar(outer, length=360, mode="determinate", maximum=100)
        self._progress_bar.pack(fill="x", pady=(0, 14))

        grid = tk.Frame(outer)
        grid.pack(fill="x", pady=(0, 12))

        self._stat_vars: dict[str, tk.StringVar] = {}
        rows = [
            ("Status",          "status",     "Starting…"),
            ("Device",          "device",     "—"),
            ("Steps done",      "steps",      "—"),
            ("Steps remaining", "remaining",  "—"),
            ("Progress",        "pct",        "0 %"),
            ("Elapsed",         "elapsed",    "—"),
            ("ETA",             "eta",        "—"),
            ("Steps / sec",     "sps",        "—"),
            ("Game score",      "eval_score", "—"),
        ]
        for i, (label, key, initial) in enumerate(rows):
            tk.Label(grid, text=label, anchor="w", width=18,
                     font=("TkDefaultFont", 10)).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=initial)
            self._stat_vars[key] = var
            tk.Label(grid, textvariable=var, anchor="w",
                     font=("TkDefaultFont", 10, "bold")).grid(row=i, column=1, sticky="w", padx=(8, 0), pady=2)

        self._chart = EvalScoreChart(
            outer, height=180,
            eval_freq=self._eval_freq,
            n_eval_episodes=self._n_eval_episodes,
        )
        self._chart.pack(fill="both", expand=True, pady=(4, 16))

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
            start_new_session=True,
        )
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _kill_process_tree(self) -> None:
        """Terminate the training process and all SubprocVecEnv worker children."""
        if not self._proc or self._proc.poll() is not None:
            return

        if sys.platform == "win32":
            self._proc.terminate()
        else:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except PermissionError:
                self._proc.terminate()

        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._proc.kill()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    def _notify_done_once(self) -> None:
        if self._done_cb_called:
            return
        self._done_cb_called = True
        self._on_done()

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
            self._chart.set_total_steps(self._total)
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

        elif t == "eval":
            steps = msg.get("steps", 0)
            mean_score = msg.get("mean_score")
            if mean_score is None:
                episodes = msg.get("episodes") or []
                if episodes:
                    mean_score = sum(ep["score"] for ep in episodes) / len(episodes)
            if mean_score is not None:
                self._stat_vars["eval_score"].set(f"{mean_score:,.0f}  @ {steps:,} steps")
                self._chart.add_point(steps, float(mean_score))

        elif t == "done":
            self._progress_bar["value"] = 100
            self._stat_vars["status"].set("Done — model saved")
            self._stat_vars["pct"].set("100.0 %")
            self._stat_vars["elapsed"].set(_fmt_seconds(msg["elapsed"]))
            self._stat_vars["eta"].set("—")
            self._stat_vars["remaining"].set("0")
            self._cancel_btn.config(text="Close", command=self._close_clean)
            self._finished = True
            self._notify_done_once()

        elif t == "_eof":
            if not self._finished:
                self._stat_vars["status"].set("Process ended")
                self._cancel_btn.config(text="Close", command=self._close_clean)
                self._finished = True
                self._notify_done_once()

    # ── close handling ────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._kill_process_tree()
            if not self._finished:
                self._stat_vars["status"].set("Cancelled")
                self._finished = True
                self._notify_done_once()
        self._close_clean()

    def _close_clean(self) -> None:
        self._finished = True
        self.destroy()
