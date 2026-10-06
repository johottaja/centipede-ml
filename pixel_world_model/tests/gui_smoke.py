"""Native GUI smoke check, kept separate from headless unit tests."""
import json
import time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from pixel_world_model.config import DEFAULTS, ROOT, save, load
from pixel_world_model.gui import Launcher


def main():
    work = ROOT / 'data' / 'verification'
    work.mkdir(parents=True, exist_ok=True)
    settings = work / 'gui_settings.json'
    save(DEFAULTS, settings)
    root = tk.Tk()
    with patch('pixel_world_model.gui.load', lambda: load(settings)), patch('pixel_world_model.gui.save', lambda values: save(values, settings)):
        launcher = Launcher(root)
        root.update()
        assert len(launcher.notebook.tabs()) == 3
        for index, title in enumerate(('Data', 'Step 1: VAE', 'Step 2: Dynamics')):
            launcher.notebook.select(index)
            root.update()
            assert launcher.notebook.tab(index, 'text') == title
        launcher.variables['dataset'].set(str(work/'rollout'))
        launcher.variables['vae_output'].set(str(work/'gui_vae'))
        launcher.variables['vae_checkpoint'].set(str(work/'gui_vae/final.pt'))
        launcher.variables['dynamics_output'].set(str(work/'dynamics'))
        launcher.variables['dynamics_checkpoint'].set(str(work/'dynamics/final.pt'))
        launcher.variables['device'].set('cpu')
        launcher.variables['vae_epochs'].set('1000')
        launcher.variables['vae_batch'].set('8')
        launcher.variables['latent_dim'].set('8')
        launcher.variables['dynamics_batch'].set('8')
        launcher.variables['hidden_width'].set('32')
        launcher.persist()
        assert load(settings)['vae_epochs'] == 1000
        launcher.launch('train-vae')
        assert launcher.job is not None
        assert all(str(b['state']) == 'disabled' for b, cmd in launcher.buttons if not cmd.startswith('inspect-'))
        start = time.monotonic()
        cancelled = False
        ticks = 0
        while launcher.job is not None and time.monotonic()-start < 40:
            root.update()
            ticks += 1
            if launcher.progress['value'] > 0 and not cancelled:
                launcher.stop(); cancelled = True
            time.sleep(0.02)
        assert cancelled, launcher.log.get('1.0', 'end')
        assert launcher.job is None
        assert (work/'gui_vae/cancelled.pt').exists()
        assert all(str(b['state']) == 'normal' for b, _ in launcher.buttons)
        assert ticks > 10
        (work/'gui_verified.json').write_text(json.dumps({'responsive_ticks': ticks, 'cancelled_checkpoint': True, 'settings_persisted': True}))
        launcher.finish_close()
    print('Native Tkinter smoke passed')


if __name__ == '__main__':
    main()
