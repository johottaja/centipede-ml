"""Standalone experimental Tkinter launcher. No imports from the main GUI."""
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pixel_world_model.config import DEFAULTS, PROJECT, SETTINGS, load, save, validate

GROUPS = {
    'Data': [('policy_model', 'C51 policy (.zip)', 'file'), ('dataset', 'Dataset directory', 'directory'),
             ('transitions', 'Transitions', ''), ('resolution', 'Image resolution', 'resolution'),
             ('exploration', 'Random action probability', ''), ('seed', 'Seed', ''),
             ('validation_fraction', 'Validation fraction', ''), ('episode_limit', 'Maximum trajectory steps', '')],
    'Step 1: VAE': [('dataset', 'Dataset directory', 'directory'), ('latent_dim', 'Latent dimension', ''),
                    ('beta', 'KL weight (beta)', ''),
                    ('vae_lr', 'Learning rate', ''),
                    ('vae_batch', 'Batch size', ''), ('vae_epochs', 'Total epochs', ''),
                    ('vae_output', 'Checkpoint output directory', 'directory'),
                    ('vae_resume', 'Resume checkpoint (optional)', 'file'),
                    ('vae_checkpoint', 'VAE checkpoint to inspect / use in Step 2', 'checkpoint')],
    'Step 2: Dynamics': [('dataset', 'Dataset directory', 'directory'),
                        ('vae_checkpoint', 'Frozen VAE checkpoint', 'checkpoint'),
                        ('dynamics_lr', 'Learning rate', ''), ('dynamics_batch', 'Batch size', ''),
                        ('dynamics_epochs', 'Total epochs', ''), ('hidden_width', 'Hidden width', ''),
                        ('blocks', 'Residual blocks', 'blocks'),
                        ('dynamics_output', 'Checkpoint output directory', 'directory'),
                        ('dynamics_resume', 'Resume checkpoint (optional)', 'file'),
                        ('dynamics_checkpoint', 'Dynamics checkpoint to inspect', 'checkpoint'),
                        ('horizon', 'Inspection horizon', '')],
}


class Launcher:
    def __init__(self, root):
        self.root = root
        root.title('Pixel world model — two-stage experiment')
        root.geometry('900x800'); root.minsize(700, 600)
        self.events = queue.Queue(); self.job = None; self.inspectors = []
        values = load()
        self.variables = {k: tk.StringVar(value=str(values[k])) for k in DEFAULTS}
        self.buttons = []
        self.notebook = None
        self.closing = False
        ttk.Label(root, text='Collect → Train VAE → Inspect → Train dynamics → Inspect', font=('TkDefaultFont', 14)).pack(anchor='w', padx=16, pady=12)
        notebook = ttk.Notebook(root)
        self.notebook = notebook
        notebook.pack(fill='both', expand=True, padx=12)
        for title, fields in GROUPS.items():
            tab = ttk.Frame(notebook); notebook.add(tab, text=title)
            canvas = tk.Canvas(tab, highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y'); canvas.pack(fill='both', expand=True)
            form = ttk.Frame(canvas, padding=12)
            window = canvas.create_window((0, 0), window=form, anchor='nw')
            form.bind('<Configure>', lambda e, view=canvas: view.configure(scrollregion=view.bbox('all')))
            canvas.bind('<Configure>', lambda e, view=canvas, win=window: view.itemconfigure(win, width=e.width))
            def wheel(event, view=canvas):
                view.yview_scroll(-int(event.delta) if abs(event.delta) < 120 else -int(event.delta / 120), 'units')
            canvas.bind('<MouseWheel>', wheel)
            form.bind('<MouseWheel>', wheel)
            form.columnconfigure(1, weight=1)
            all_fields = fields + [('device', 'Device', 'device'), ('checkpoint_frequency', 'Checkpoint frequency (epochs)', '')]
            for row, (key, label, kind) in enumerate(all_fields):
                ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=5)
                if kind in ('resolution', 'blocks', 'device'):
                    choices = {'resolution': (64, 84, 128, 192, 256), 'blocks': (2, 3), 'device': ('auto', 'cpu', 'mps', 'cuda')}[kind]
                    widget = ttk.Combobox(form, textvariable=self.variables[key], values=choices, state='readonly')
                elif kind == 'checkpoint':
                    paths = sorted((PROJECT / 'pixel_world_model' / 'checkpoints').rglob('*.pt'))
                    widget = ttk.Combobox(form, textvariable=self.variables[key], values=[str(p.relative_to(PROJECT)) for p in paths])
                    widget.configure(postcommand=lambda w=widget: w.configure(values=[str(p.relative_to(PROJECT)) for p in sorted((PROJECT/'pixel_world_model/checkpoints').rglob('*.pt'))]))
                else:
                    widget = ttk.Entry(form, textvariable=self.variables[key])
                widget.grid(row=row, column=1, sticky='ew', pady=5)
                widget.bind('<MouseWheel>', wheel)
                if kind in ('file', 'checkpoint', 'directory'):
                    ttk.Button(form, text='Browse…', command=lambda k=key, d=kind: self.browse(k, d)).grid(row=row, column=2, padx=6)
            bar = ttk.Frame(tab, padding=10); bar.pack(fill='x')
            if title == 'Data':
                self.button(bar, 'Collect dataset', 'collect')
            elif title == 'Step 1: VAE':
                self.button(bar, 'Train Step 1', 'train-vae'); self.button(bar, 'Inspect VAE', 'inspect-vae')
            else:
                self.button(bar, 'Train Step 2', 'train-dynamics'); self.button(bar, 'Inspect dynamics', 'inspect-dynamics')
        bar = ttk.Frame(root, padding=12); bar.pack(fill='x')
        ttk.Button(bar, text='Save settings', command=self.persist).pack(side='left')
        self.cancel = ttk.Button(bar, text='Cancel job', command=self.stop, state='disabled'); self.cancel.pack(side='right')
        self.status = tk.StringVar(value='Ready. Select a policy and collect a dataset first.')
        ttk.Label(root, textvariable=self.status, wraplength=850).pack(anchor='w', padx=14)
        self.stats = tk.StringVar(value='Device: — | Epoch: — | Updates: — | Validation: —')
        ttk.Label(root, textvariable=self.stats, wraplength=850).pack(anchor='w', padx=14)
        self.progress = ttk.Progressbar(root, maximum=100); self.progress.pack(fill='x', padx=14, pady=6)
        self.log = tk.Text(root, height=9, state='disabled', wrap='word'); self.log.pack(fill='x', padx=14, pady=(0, 12))
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def button(self, bar, label, command):
        button = ttk.Button(bar, text=label, command=lambda: self.launch(command))
        button.pack(side='left', padx=4)
        self.buttons.append((button, command))

    def browse(self, key, kind):
        value = filedialog.askdirectory(parent=self.root) if kind == 'directory' else filedialog.askopenfilename(parent=self.root, filetypes=[('Models', '*.pt *.zip'), ('All files', '*')])
        if value:
            self.variables[key].set(value)

    def configuration(self):
        c = {}
        for key, default in DEFAULTS.items():
            try:
                c[key] = type(default)(self.variables[key].get())
            except ValueError:
                raise ValueError(f'{key}: enter a valid {type(default).__name__}') from None
        validate(c)
        return c

    def persist(self):
        try:
            c = self.configuration(); save(c); self.status.set(f'Settings saved to {SETTINGS}')
            return c
        except Exception as e:
            self.status.set(str(e)); messagebox.showerror('Configuration error', str(e), parent=self.root)
            return None

    def launch(self, command):
        c = self.persist()
        if c is None:
            return
        inspection = command.startswith('inspect-')
        if self.job is not None and not inspection:
            return
        args = [sys.executable, '-u', '-m', 'pixel_world_model.cli', command]
        # Pass a snapshot so an open form cannot change an already-running job.
        for key, value in c.items():
            args.extend(['--'+key.replace('_', '-'), str(value)])
        try:
            proc = subprocess.Popen(args, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as exc:
            messagebox.showerror('Cannot launch job', str(exc), parent=self.root)
            return
        if inspection:
            self.inspectors.append(proc)
        else:
            self.job = proc
            self.progress['value'] = 0
            self.cancel['state'] = 'normal'
            for button, cmd in self.buttons:
                if not cmd.startswith('inspect-'):
                    button['state'] = 'disabled'
        self.status.set(f'Running {command}')
        for stream, structured in ((proc.stdout, True), (proc.stderr, False)):
            threading.Thread(target=self.read, args=(proc, stream, structured), daemon=True).start()

    def read(self, proc, stream, structured):
        for line in stream:
            try:
                event = json.loads(line) if structured else {'type': 'log', 'text': line.rstrip()}
                if not isinstance(event, dict) or 'type' not in event:
                    event = {'type': 'log', 'text': line.rstrip()}
            except json.JSONDecodeError:
                event = {'type': 'log', 'text': line.rstrip()}
            self.events.put((proc, event))
        stream.close()

    def stop(self):
        if self.job and self.job.poll() is None:
            self.job.terminate()
            self.status.set('Cancelling after current update; saving resumable checkpoint…')
            self.cancel['state'] = 'disabled'

    def poll(self):
        while not self.events.empty():
            proc, event = self.events.get()
            if proc == self.job and event['type'] == 'device':
                self.stats.set(f"Device: {event['device']}")
            if proc == self.job and event['type'] == 'validation':
                self.stats.set(f"Epoch: {event['epoch']} | Validation: " + ' | '.join(f'{"BCE" if k == "reconstruction" else k}: {v:.6g}' for k, v in event.items() if k not in ('type', 'epoch')))
            if event['type'] == 'progress' and proc == self.job:
                self.progress['value'] = 100*event['current']/max(event['total'], 1)
                if 'epoch' in event:
                    self.stats.set(f"Epoch: {event['epoch']} | Updates: {event['updates']} | " + ' | '.join(f'{"BCE" if k == "reconstruction" else k}: {event[k]:.6g}' for k in ('reconstruction', 'pixel_mse', 'kl', 'latent_mse') if k in event))
                self.status.set(f"{event.get('phase', '')}: {event['current']}/{event['total']} | elapsed {event['elapsed']:.0f}s | ETA {event['eta']:.0f}s | loss {event.get('loss', '—')}")
            text = json.dumps(event) if event['type'] != 'log' else event['text']
            self.log.configure(state='normal'); self.log.insert('end', text+'\n')
            if int(self.log.index('end-1c').split('.')[0]) > 500:
                self.log.delete('1.0', '100.0')
            self.log.see('end'); self.log.configure(state='disabled')
            if event['type'] == 'error':
                self.status.set(event['text'])
            if event['type'] == 'done' and proc == self.job:
                if event.get('cancelled'):
                    self.status.set('Cancelled before training; latent cache can be rebuilt.' if event.get('checkpoint_saved') is False else 'Cancelled; see saved checkpoint in log.')
                else:
                    self.status.set('Complete. Inspect results before starting the next stage.')
        if self.job is not None and self.job.poll() is not None:
            code = self.job.returncode
            self.job = None; self.cancel['state'] = 'disabled'
            for button, _ in self.buttons:
                button['state'] = 'normal'
            if code:
                self.status.set('Job failed. See the error and diagnostics below.')
        self.inspectors = [p for p in self.inspectors if p.poll() is None]
        self.root.after(100, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.job is not None:
            self.stop()
            self.status.set('Waiting for checkpoint save before closing…')
            self.root.after(200, self.close_when_done)
        else:
            self.finish_close()

    def close_when_done(self):
        if self.job is not None and self.job.poll() is None:
            self.root.after(200, self.close_when_done)
        else:
            self.finish_close()

    def finish_close(self):
        for proc in self.inspectors:
            if proc.poll() is None:
                proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk(); Launcher(root); root.mainloop()
