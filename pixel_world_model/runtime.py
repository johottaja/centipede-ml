import hashlib
import json
import signal
import time
from pathlib import Path
import torch

VERSION = 2


def emit(kind, **fields):
    print(json.dumps({'type': kind, **fields}), flush=True)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def device_for(name='auto'):
    if name != 'auto':
        if name == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; choose auto or cpu.')
        if name == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; choose auto or cpu.')
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class Cancellation:
    def __init__(self):
        self.cancelled = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop)

    def stop(self, *_):
        self.cancelled = True


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, stage, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('version') != VERSION or ck.get('stage') != stage:
        raise ValueError(f'{path}: incompatible checkpoint. Select a new {stage} checkpoint; legacy joint models cannot be converted.')
    return ck


def progress(start, current, total, initial=0, **fields):
    elapsed = time.monotonic() - start
    emit('progress', current=current, total=total, elapsed=elapsed,
         eta=elapsed / max(current-initial, 1) * (total-current), **fields)
