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


def _fmt_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M"
    if steps >= 1_000:
        return f"{steps / 1_000:.0f}k"
    return str(steps)


class EvalScoreChart(tk.Canvas):
    """Simple line chart for eval mean scores over training steps."""

    _MARGIN_LEFT = 52
    _MARGIN_RIGHT = 12
    _MARGIN_TOP = 16
    _MARGIN_BOTTOM = 28

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
        self._points: list[tuple[int, float]] = []
        self.bind("<Configure>", self._on_resize)

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
            text=f"Eval score ({self._n_eval_episodes}-game avg, deterministic)",
            font=("TkDefaultFont", 9, "bold"),
            fill="#333333",
        )

        if not self._points:
            self.create_text(
                w // 2, (plot_t + plot_b) // 2,
                text=f"Waiting for first evaluation at {_fmt_steps(self._eval_freq)} steps…",
                font=("TkDefaultFont", 9),
                fill="#888888",
            )
            return

        scores = [p[1] for p in self._points]
        y_min = min(scores)
        y_max = max(scores)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        pad = (y_max - y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad

        steps_min = self._points[0][0]
        steps_max = self._points[-1][0]
        if steps_min == steps_max:
            steps_max = steps_min + 1

        def x_pos(steps: int) -> float:
            return plot_l + (steps - steps_min) / (steps_max - steps_min) * plot_w

        def y_pos(score: float) -> float:
            return plot_b - (score - y_min) / (y_max - y_min) * plot_h

        # Axes
        self.create_line(plot_l, plot_t, plot_l, plot_b, fill="#999999")
        self.create_line(plot_l, plot_b, plot_r, plot_b, fill="#999999")

        # Y-axis labels (min, mid, max)
        for val in (y_min, (y_min + y_max) / 2, y_max):
            y = y_pos(val)
            self.create_line(plot_l - 3, y, plot_l, y, fill="#999999")
            label = f"{val:,.0f}" if abs(val) >= 10 else f"{val:.1f}"
            self.create_text(plot_l - 6, y, text=label, anchor="e", font=("TkDefaultFont", 8), fill="#666666")

        # X-axis labels (first, last step)
        for steps in (steps_min, steps_max):
            x = x_pos(steps)
            self.create_line(x, plot_b, x, plot_b + 3, fill="#999999")
            self.create_text(x, plot_b + 10, text=_fmt_steps(steps), font=("TkDefaultFont", 8), fill="#666666")

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
            ("Eval score",      "eval_score", "—"),
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

        elif t == "eval":
            mean_score = msg.get("mean_score")
            steps = msg.get("steps", 0)
            if mean_score is not None:
                self._stat_vars["eval_score"].set(f"{mean_score:,.1f}  @ {steps:,} steps")
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
