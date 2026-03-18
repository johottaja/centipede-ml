import tkinter as tk
from tkinter import ttk

from .tooltip import make_tooltip

# (label, cli-key, default, type, min, max, tooltip)
HPARAM_SPECS: list[tuple] = [
    ("Timesteps",              "timesteps",              "1000000", "int",   1,    None, "Total environment steps to train for"),
    ("Seed",                   "seed",                   "0",       "int",   0,    None, "Random seed for reproducibility"),
    ("Learning rate",          "learning-rate",          "0.0001",  "float", 0,    1,    "Adam optimizer learning rate"),
    ("Buffer size",            "buffer-size",            "100000",  "int",   1,    None, "Replay buffer capacity"),
    ("Learning starts",        "learning-starts",        "10000",   "int",   0,    None, "Steps before first gradient update"),
    ("Batch size",             "batch-size",             "64",      "int",   1,    None, "Mini-batch size for each update"),
    ("Tau (soft update)",      "tau",                    "1.0",     "float", 0,    1,    "Soft update coefficient for target network (1 = hard copy)"),
    ("Gamma (discount)",       "gamma",                  "0.99",    "float", 0,    1,    "Discount factor for future rewards"),
    ("Train frequency",        "train-freq",             "4",       "int",   1,    None, "Steps between gradient updates"),
    ("Gradient steps",         "gradient-steps",         "1",       "int",   1,    None, "Gradient updates per train call"),
    ("Target update interval", "target-update-interval", "1000",    "int",   1,    None, "Steps between target network syncs"),
    ("Exploration fraction",   "exploration-fraction",   "0.1",     "float", 0,    1,    "Fraction of training spent exploring"),
    ("Final epsilon",          "exploration-final-eps",  "0.01",    "float", 0,    1,    "Epsilon at end of exploration schedule"),
    ("Net architecture",       "net-arch",               "256,256", "str",   None, None, "Hidden layer sizes, comma-separated (e.g. 256,256)"),
]


class HParamPanel(tk.Frame):
    """Scrollable grid of labelled entry widgets for each hyperparameter."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self._vars: dict[str, tk.StringVar] = {}
        self._build()

    def _build(self) -> None:
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: (
            canvas.configure(scrollregion=canvas.bbox("all")),
            canvas.itemconfig(inner_id, width=canvas.winfo_width()),
        ))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        for row, (label, key, default, _type, _min, _max, tooltip) in enumerate(HPARAM_SPECS):
            var = tk.StringVar(value=default)
            self._vars[key] = var

            lbl = tk.Label(inner, text=label, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", padx=(4, 8), pady=3)
            make_tooltip(lbl, tooltip)

            entry = tk.Entry(inner, textvariable=var, width=18)
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 4), pady=3)
            make_tooltip(entry, tooltip)

        inner.columnconfigure(1, weight=1)

    def reset_defaults(self) -> None:
        for _label, key, default, *_ in HPARAM_SPECS:
            self._vars[key].set(default)

    def get_args(self) -> list[str]:
        """Validate all entries and return a flat CLI arg list for train.py.

        Raises ValueError with a newline-separated message on any invalid input.
        """
        args: list[str] = []
        errors: list[str] = []

        for label, key, _default, typ, mn, mx, _ in HPARAM_SPECS:
            raw = self._vars[key].get().strip()
            if not raw:
                errors.append(f"{label}: must not be empty")
                continue

            if typ == "int":
                try:
                    val = int(raw)
                except ValueError:
                    errors.append(f"{label}: must be an integer")
                    continue
                if mn is not None and val < mn:
                    errors.append(f"{label}: must be ≥ {mn}")
                    continue
            elif typ == "float":
                try:
                    val = float(raw)
                except ValueError:
                    errors.append(f"{label}: must be a number")
                    continue
                if mn is not None and val < mn:
                    errors.append(f"{label}: must be ≥ {mn}")
                    continue
                if mx is not None and val > mx:
                    errors.append(f"{label}: must be ≤ {mx}")
                    continue
            elif typ == "str":
                parts = [p.strip() for p in raw.split(",")]
                if not all(p.isdigit() and int(p) > 0 for p in parts):
                    errors.append(f"{label}: must be comma-separated positive integers (e.g. 256,256)")
                    continue

            args += [f"--{key}", raw]

        if errors:
            raise ValueError("\n".join(errors))

        return args
