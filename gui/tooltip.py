import tkinter as tk


def make_tooltip(widget: tk.Widget, text: str) -> None:
    """Attach a hover tooltip to *widget* showing *text*."""
    tip_window: list[tk.Toplevel | None] = [None]

    def show(event: tk.Event) -> None:
        if tip_window[0]:
            return
        x = widget.winfo_rootx() + widget.winfo_width() + 4
        y = widget.winfo_rooty()
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), wraplength=260, padx=4, pady=2,
        ).pack()
        tip_window[0] = tw

    def hide(event: tk.Event) -> None:
        if tip_window[0]:
            tip_window[0].destroy()
            tip_window[0] = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)
