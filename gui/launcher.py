import datetime
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

from .hparam_panel import HParamPanel
from .progress_window import ProgressWindow

MODEL_ZIP = os.path.join("models", "dqn_centipede.zip")


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Centipede")
        self.resizable(True, True)
        self.minsize(460, 520)

        self._hparam_panel: HParamPanel
        self._progress_win: ProgressWindow | None = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        tk.Label(self, text="Centipede", font=("TkDefaultFont", 16, "bold")).pack(pady=(16, 8))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        nb.add(self._build_play_tab(nb),  text="  Play  ")
        nb.add(self._build_train_tab(nb), text="  Train Agent  ")
        nb.add(self._build_watch_tab(nb), text="  Watch Agent  ")

        self._build_status_bar()

    def _build_play_tab(self, parent: ttk.Notebook) -> tk.Frame:
        tab = tk.Frame(parent, padx=20, pady=20)
        tk.Label(tab, text="Play Centipede yourself", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 12))
        tk.Button(tab, text="Launch Game", width=24, command=self._launch_game).pack(pady=4)
        return tab

    def _build_train_tab(self, parent: ttk.Notebook) -> tk.Frame:
        tab = tk.Frame(parent, padx=12, pady=12)
        tk.Label(tab, text="DQN Hyperparameters", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 6))
        tk.Label(tab, text="Hover over any field for a description.",
                 fg="gray", font=("TkDefaultFont", 9)).pack(pady=(0, 8))

        self._hparam_panel = HParamPanel(tab)
        self._hparam_panel.pack(fill="both", expand=True)

        btn_row = tk.Frame(tab)
        btn_row.pack(fill="x", pady=(10, 0))
        self._train_btn = tk.Button(btn_row, text="Start Training", width=20, command=self._launch_train)
        self._train_btn.pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Reset Defaults", width=16, command=self._hparam_panel.reset_defaults).pack(side="left")

        return tab

    def _build_watch_tab(self, parent: ttk.Notebook) -> tk.Frame:
        tab = tk.Frame(parent, padx=20, pady=20)
        tk.Label(tab, text="Watch the trained agent play", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 12))
        tk.Button(tab, text="Watch Agent", width=24, command=self._launch_watch).pack(pady=4)
        return tab

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, padx=12, pady=4)
        bar.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar(value=self._model_status())
        tk.Label(bar, textvariable=self._status_var, fg="gray",
                 font=("TkDefaultFont", 9), anchor="w").pack(side="left")
        tk.Button(bar, text="Quit", command=self.destroy).pack(side="right")

    # ── helpers ───────────────────────────────────────────────────────────

    def _model_status(self) -> str:
        if os.path.exists(MODEL_ZIP):
            mtime = os.path.getmtime(MODEL_ZIP)
            dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            return f"Model found (saved {dt})"
        return "No trained model yet"

    def _refresh_status(self) -> None:
        self._status_var.set(self._model_status())

    # ── actions ───────────────────────────────────────────────────────────

    def _launch_game(self) -> None:
        import subprocess
        subprocess.Popen([sys.executable, "play.py"])

    def _launch_train(self) -> None:
        if self._progress_win and self._progress_win.winfo_exists():
            self._progress_win.lift()
            return
        try:
            extra_args = self._hparam_panel.get_args()
        except ValueError as exc:
            messagebox.showerror("Invalid hyperparameters", str(exc))
            return
        cmd = [sys.executable, "train.py"] + extra_args
        self._train_btn.config(text="Training… (running)", state="disabled")
        self._progress_win = ProgressWindow(self, cmd, on_done=self._on_train_done)

    def _on_train_done(self) -> None:
        self._train_btn.config(text="Start Training", state="normal")
        self._refresh_status()

    def _launch_watch(self) -> None:
        import subprocess
        if not os.path.exists(MODEL_ZIP):
            messagebox.showwarning("No model", "Train the agent first — no model file found.")
            return
        subprocess.Popen([sys.executable, "watch.py"])
