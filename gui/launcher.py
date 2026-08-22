import datetime
import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

from .hparam_panel import HParamPanel, SETTINGS_PATH
from .progress_window import ProgressWindow
from core.train import MODEL_PATH, list_saved_models

MODEL_ZIP = MODEL_PATH + ".zip"


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Centipede")
        self.resizable(True, True)
        self.minsize(460, 520)

        self._hparam_panel: HParamPanel
        self._progress_win: ProgressWindow | None = None
        self._watch_model_var = tk.StringVar()
        self._watch_models: dict[str, str] = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        tk.Label(self, text="Centipede", font=("TkDefaultFont", 16, "bold")).pack(pady=(16, 8))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        nb.add(self._build_play_tab(nb),  text="  Play  ")
        nb.add(self._build_train_tab(nb), text="  Train Agent  ")
        nb.add(self._build_watch_tab(nb), text="  Watch Agent  ")
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_status_bar()

    def _build_play_tab(self, parent: ttk.Notebook) -> tk.Frame:
        tab = tk.Frame(parent, padx=20, pady=20)
        tk.Label(tab, text="Play Centipede yourself", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 12))
        tk.Button(tab, text="Launch Game", width=24, command=self._launch_game).pack(pady=4)
        return tab

    def _build_train_tab(self, parent: ttk.Notebook) -> tk.Frame:
        tab = tk.Frame(parent, padx=12, pady=12)
        tk.Label(tab, text="DDQN / CNN Hyperparameters", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 6))
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

        picker = tk.Frame(tab)
        picker.pack(fill="x", pady=(0, 8))
        tk.Label(picker, text="Model:", anchor="w").pack(side="left")
        self._watch_combo = ttk.Combobox(
            picker,
            textvariable=self._watch_model_var,
            state="readonly",
            width=36,
        )
        self._watch_combo.pack(side="left", padx=(8, 0), fill="x", expand=True)

        btn_row = tk.Frame(tab)
        btn_row.pack(fill="x", pady=(8, 0))
        tk.Button(btn_row, text="Refresh list", width=14, command=self._refresh_watch_models).pack(side="left", padx=(0, 8))
        self._watch_btn = tk.Button(btn_row, text="Watch Agent", width=14, command=self._launch_watch)
        self._watch_btn.pack(side="left")

        self._refresh_watch_models()
        return tab

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, padx=12, pady=4)
        bar.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar(value=self._model_status())
        tk.Label(bar, textvariable=self._status_var, fg="gray",
                 font=("TkDefaultFont", 9), anchor="w").pack(side="left")
        tk.Button(bar, text="Quit", command=self.destroy).pack(side="right")

    # ── helpers ───────────────────────────────────────────────────────────

    def _load_settings(self) -> dict:
        if not os.path.exists(SETTINGS_PATH):
            return {}
        try:
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_watch_model_pref(self, path: str) -> None:
        data = self._load_settings()
        data["watch-model"] = path
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _refresh_watch_models(self) -> None:
        models = list_saved_models()
        self._watch_models = {label: path for path, label in models}
        labels = [label for _, label in models]
        self._watch_combo["values"] = labels

        saved = self._load_settings().get("watch-model", "")
        selected = next((label for label, path in models if path == saved), None)
        if selected:
            self._watch_model_var.set(selected)
        elif labels:
            self._watch_combo.current(0)
        else:
            self._watch_model_var.set("")

        state = "normal" if labels else "disabled"
        self._watch_btn.config(state=state)
        self._watch_combo.config(state="readonly" if labels else "disabled")

    def _model_status(self) -> str:
        models = list_saved_models()
        if not models:
            return "No trained model yet"
        if os.path.exists(MODEL_ZIP):
            mtime = os.path.getmtime(MODEL_ZIP)
            dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            extra = len(models) - 1
            if extra > 0:
                return f"Final model saved {dt} (+{extra} checkpoint{'s' if extra != 1 else ''})"
            return f"Model found (saved {dt})"
        return f"{len(models)} checkpoint{'s' if len(models) != 1 else ''} available"

    def _refresh_status(self) -> None:
        self._status_var.set(self._model_status())
        self._refresh_watch_models()

    def _on_tab_changed(self, event: tk.Event) -> None:
        tab_text = event.widget.tab(event.widget.select(), "text").strip()
        if tab_text == "Watch Agent":
            self._refresh_watch_models()

    # ── actions ───────────────────────────────────────────────────────────

    def _launch_game(self) -> None:
        import subprocess
        subprocess.Popen([sys.executable, "-m", "core.play"])

    def _launch_train(self) -> None:
        if self._progress_win and self._progress_win.winfo_exists():
            self._progress_win.lift()
            return
        try:
            extra_args = self._hparam_panel.get_args()
        except ValueError as exc:
            messagebox.showerror("Invalid hyperparameters", str(exc))
            return
        self._hparam_panel.save()
        cmd = [sys.executable, "-m", "core.train"] + extra_args
        eval_freq = int(self._hparam_panel._vars["eval-freq"].get())
        self._train_btn.config(text="Training… (running)", state="disabled")
        self._progress_win = ProgressWindow(self, cmd, on_done=self._on_train_done, eval_freq=eval_freq)

    def _on_train_done(self) -> None:
        self._train_btn.config(text="Start Training", state="normal")
        self._refresh_status()

    def _launch_watch(self) -> None:
        import subprocess
        label = self._watch_model_var.get()
        model_path = self._watch_models.get(label)
        if not model_path:
            messagebox.showwarning("No model", "Select a model to watch, or train the agent first.")
            return
        self._save_watch_model_pref(model_path)
        subprocess.Popen([sys.executable, "-m", "core.watch", "--model", model_path])
