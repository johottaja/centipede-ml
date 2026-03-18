import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

MODEL_ZIP = os.path.join("models", "dqn_centipede.zip")


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Centipede")
        self.resizable(False, False)

        frame = tk.Frame(self, padx=20, pady=20)
        frame.pack()

        tk.Label(frame, text="Centipede", font=("TkDefaultFont", 16, "bold")).pack(pady=(0, 16))

        tk.Button(frame, text="Play", width=24, command=self._launch_game).pack(pady=4)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(frame, text="DQN Agent", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 6))

        self._train_btn = tk.Button(
            frame, text="Train Agent", width=24, command=self._launch_train
        )
        self._train_btn.pack(pady=4)

        self._watch_btn = tk.Button(
            frame, text="Watch Agent", width=24, command=self._launch_watch
        )
        self._watch_btn.pack(pady=4)

        self._status_var = tk.StringVar(value=self._model_status())
        tk.Label(frame, textvariable=self._status_var, fg="gray", font=("TkDefaultFont", 9)).pack(
            pady=(6, 0)
        )

        tk.Button(frame, text="Quit", width=24, command=self.destroy).pack(pady=(12, 0))

        self._train_proc: subprocess.Popen | None = None
        self._poll_train()

    # ------------------------------------------------------------------
    def _model_status(self) -> str:
        if os.path.exists(MODEL_ZIP):
            mtime = os.path.getmtime(MODEL_ZIP)
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            return f"Model found (saved {dt})"
        return "No trained model yet"

    def _refresh_status(self):
        self._status_var.set(self._model_status())

    # ------------------------------------------------------------------
    def _launch_game(self):
        subprocess.Popen([sys.executable, "play.py"])

    def _launch_train(self):
        if self._train_proc and self._train_proc.poll() is None:
            messagebox.showinfo("Training", "Training is already running.")
            return
        self._train_proc = subprocess.Popen([sys.executable, "train.py"])
        self._train_btn.config(text="Training… (running)", state="disabled")
        self._poll_train()

    def _poll_train(self):
        if self._train_proc and self._train_proc.poll() is not None:
            self._train_btn.config(text="Train Agent", state="normal")
            self._refresh_status()
            self._train_proc = None
        self.after(1000, self._poll_train)

    def _launch_watch(self):
        if not os.path.exists(MODEL_ZIP):
            messagebox.showwarning("No model", "Train the agent first — no model file found.")
            return
        subprocess.Popen([sys.executable, "watch.py"])


if __name__ == "__main__":
    Launcher().mainloop()
