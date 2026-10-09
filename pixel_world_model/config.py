"""Independent experiment configuration; paths are relative to the project root."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SETTINGS = ROOT / 'settings.json'
DEFAULTS = {
    'policy_model': 'models/dqn_centipede', 'dataset': 'pixel_world_model/data/diverse',
    'collection_mode': 'sampled', 'collection_workers': 4, 'episodes': 500, 'validation_episodes': 50, 'samples_per_episode': 100, 'resolution': 128, 'exploration': 0.1, 'seed': 0,
    'episode_limit': 10000,
    'latent_dim': 256, 'beta': 0.0001, 'vae_foreground_weight': 9.0, 'vae_lr': 0.0001, 'vae_batch': 64,
    'vae_epochs': 50, 'dynamics_lr': 0.0001, 'dynamics_batch': 256,
    'dynamics_foreground_weight': 9.0, 'dynamics_pixel_loss_weight': 1.0, 'dynamics_visual_batch': 16,
    'dynamics_epochs': 100, 'hidden_width': 256, 'blocks': 3,
    'checkpoint_frequency': 5, 'device': 'auto', 'horizon': 15,
    'vae_checkpoint': 'pixel_world_model/checkpoints/vae/best.pt',
    'dynamics_checkpoint': 'pixel_world_model/checkpoints/dynamics/best.pt',
    'vae_output': 'pixel_world_model/checkpoints/vae',
    'dynamics_output': 'pixel_world_model/checkpoints/dynamics',
    'vae_resume': '', 'dynamics_resume': '',
}


def load(path=SETTINGS):
    values = DEFAULTS.copy()
    if Path(path).exists():
        values.update({k: v for k, v in json.loads(Path(path).read_text()).items() if k in DEFAULTS})
    return values


def save(values, path=SETTINGS):
    Path(path).write_text(json.dumps(values, indent=2) + '\n')


def validate(c):
    for key, default in DEFAULTS.items():
        if isinstance(default, (int, float)) and not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
    if c['seed'] < 0:
        raise ValueError('seed must be nonnegative')
    if c['collection_mode'] not in ('sampled', 'consecutive'):
        raise ValueError('collection_mode must be sampled or consecutive')
    if c['device'] not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('device must be auto, cpu, mps, or cuda')
    if c['horizon'] > 100:
        raise ValueError('horizon must be between 1 and 100')
    for key in ('collection_workers', 'episodes', 'samples_per_episode', 'episode_limit', 'latent_dim', 'vae_batch', 'vae_epochs',
                'dynamics_batch', 'dynamics_visual_batch', 'dynamics_epochs', 'hidden_width', 'checkpoint_frequency', 'horizon'):
        if int(c[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if c['resolution'] not in (64, 84, 128, 192, 256):
        raise ValueError('resolution must be 64, 84, 128, 192, or 256')
    if not 1 <= c['validation_episodes'] < c['episodes']:
        raise ValueError('validation_episodes must be at least 1 and smaller than episodes')
    if not 0 <= c['exploration'] <= 1 or c['beta'] < 0:
        raise ValueError('exploration must be in [0,1]; beta must be nonnegative')
    if c['vae_foreground_weight'] <= 0:
        raise ValueError('vae_foreground_weight must be positive')
    if c['dynamics_foreground_weight'] <= 0:
        raise ValueError('dynamics_foreground_weight must be positive')
    if c['dynamics_pixel_loss_weight'] < 0:
        raise ValueError('dynamics_pixel_loss_weight must be nonnegative')
    if c['blocks'] not in (2, 3):
        raise ValueError('blocks must be 2 or 3')
    if c['vae_lr'] <= 0 or c['dynamics_lr'] <= 0:
        raise ValueError('learning rates must be positive')
