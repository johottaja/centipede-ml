"""Browser inspector for held-out VAE reconstructions and dynamics rollouts."""
import base64
import json
from pathlib import Path
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import zlib

import numpy as np

from pixel_world_model.data import TransitionDataset
from pixel_world_model.visualize import Inspection


PAGE = Path(__file__).with_name('web_inspector.html')
MAX_REQUEST_BYTES = 4096


def _png_data_url(frame):
    """Encode one grayscale panel as PNG using only the standard library."""
    pixels = np.clip(np.rint(np.asarray(frame) * 255), 0, 255).astype(np.uint8)
    height, width = pixels.shape
    raw = b''.join(b'\0' + pixels[row].tobytes() for row in range(height))

    def chunk(kind, payload):
        body = kind + payload
        return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)

    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 0, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6))
           + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


class InspectorSession:
    def __init__(self, config, stage):
        self.config = config
        self.stage = stage
        self.lock = threading.RLock()
        self.checkpoint = Path(config[f'{stage}_checkpoint'])
        self.checkpoint_signature = self._signature()
        self.view = Inspection(config, stage)
        self.revision = 0
        self.cached_state = None

    def _signature(self):
        try:
            stat = self.checkpoint.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _refresh_checkpoint(self):
        signature = self._signature()
        if signature is None or signature == self.checkpoint_signature:
            return
        previous = self.view
        view = Inspection(self.config, self.stage)
        view.position = min(previous.position, len(view.data)-1)
        view.channel = previous.channel
        view.horizon = previous.horizon
        view.prepare()
        if self.stage == 'dynamics':
            view.step = min(previous.step, len(view.actions))
        self.view = view
        self.checkpoint_signature = signature
        self.revision += 1
        self.cached_state = None

    def state(self):
        with self.lock:
            self._refresh_checkpoint()
            if self.cached_state is None:
                view = self.view
                panels, metrics = view.panels()
                episode = int(view.data.transitions['episodes'][view.index])
                modified = self.checkpoint_signature[0] / 1e9 if self.checkpoint_signature else None
                self.cached_state = {
                    'stage': self.stage,
                    'revision': self.revision,
                    'sample': view.position,
                    'samples': len(view.data),
                    'channel': view.channel,
                    'episode': episode,
                    'step': view.step,
                    'steps': len(view.actions) if self.stage == 'dynamics' else 0,
                    'horizon': view.horizon,
                    'metrics': metrics,
                    'checkpoint': self.checkpoint.name,
                    'checkpoint_modified': modified,
                    'panels': [{'title': title, 'image': _png_data_url(frame)}
                               for title, frame in panels],
                }
            return self.cached_state

    def act(self, payload):
        with self.lock:
            self._refresh_checkpoint()
            view = self.view
            action = payload.get('action')
            if action == 'advance':
                amount = payload.get('amount')
                if type(amount) is not int or amount not in (-1, 1):
                    raise ValueError('amount must be -1 or 1')
                view.advance(amount)
            elif action == 'sample':
                position = payload.get('position')
                if type(position) is not int or not 0 <= position < len(view.data):
                    raise ValueError('sample position is out of range')
                if position != view.position:
                    view.position = position
                    view.prepare()
            elif action == 'step' and self.stage == 'dynamics':
                step = payload.get('step')
                if type(step) is not int or not 0 <= step <= len(view.actions):
                    raise ValueError('rollout step is out of range')
                view.step = step
            elif action == 'channel':
                channel = payload.get('channel')
                if type(channel) is not int or not 0 <= channel < 4:
                    raise ValueError('channel must be 0 through 3')
                view.channel = channel
            elif action == 'horizon':
                horizon = payload.get('horizon')
                if type(horizon) is not int or not 1 <= horizon <= 100:
                    raise ValueError('horizon must be 1 through 100')
                if horizon != view.horizon:
                    view.horizon = horizon
                    view.prepare()
            elif action == 'next_trajectory':
                view.next_trajectory()
            elif action == 'reset':
                view.prepare()
            else:
                raise ValueError('unknown inspector action')
            self.revision += 1
            self.cached_state = None
            return self.state()


class LossHistory:
    """Read trainer scalars and expose the compact loss history used by the page."""
    def __init__(self, config, stage, logdir=None):
        self.stage = stage
        prefix = 'vae' if stage == 'vae' else 'dynamics'
        checkpoint_dir = Path(config[f'{stage}_checkpoint']).parent
        self.explicit_logdir = Path(logdir) if logdir else None
        candidates = [checkpoint_dir / 'tensorboard', Path(config[f'{prefix}_output']) / 'tensorboard']
        self.logdirs = ([self.explicit_logdir] if self.explicit_logdir else []) + list(dict.fromkeys(candidates))
        self.logdir = self.logdirs[0]
        self.train_count = len(TransitionDataset(config['dataset'], 'train'))
        self.epochs = int(config[f'{prefix}_epochs'])
        self.metric_label = 'BCE + β KL' if stage == 'vae' else 'Latent MSE + weighted prediction BCE'
        self.accumulator = None
        self.accumulator_dir = None
        self.cache = {'available': False, 'train': [], 'validation': [], 'max_epochs': self.epochs,
                      'metric_label': self.metric_label, 'updated_at': None}
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            self.logdir = (self.explicit_logdir if self.explicit_logdir else
                           next((path for path in self.logdirs
                                 if path.is_dir() and any(path.glob('events.out.tfevents.*'))), self.logdirs[0]))
            event_files = list(self.logdir.glob('events.out.tfevents.*')) if self.logdir.is_dir() else []
            if not event_files:
                return self.cache
            try:
                run_config = self.logdir.parent / 'run_config.json'
                if run_config.exists():
                    saved = json.loads(run_config.read_text())
                    prefix = 'vae' if self.stage == 'vae' else 'dynamics'
                    self.epochs = int(saved.get(f'{prefix}_epochs', self.epochs))
                if self.accumulator is None or self.accumulator_dir != self.logdir:
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    self.accumulator = EventAccumulator(str(self.logdir), size_guidance={'scalars': 0})
                    self.accumulator_dir = self.logdir
                self.accumulator.Reload()
                tags = self.accumulator.Tags().get('scalars', [])

                def series(tag):
                    if tag not in tags:
                        return []
                    return [[event.step / max(self.train_count, 1), event.value]
                            for event in self.accumulator.Scalars(tag)]

                train = series('loss/train')
                validation = series('loss/validation')
                if train or validation:
                    self.cache = {'available': True, 'train': train, 'validation': validation,
                                  'max_epochs': self.epochs, 'metric_label': self.metric_label,
                                  'updated_at': time.time()}
            except Exception:
                # Keep the last complete chart if an event file is being appended during a read.
                pass
            return self.cache


def serve(config, stage, host='127.0.0.1', port=8765, logdir=None):
    if stage not in ('vae', 'dynamics'):
        raise ValueError('stage must be vae or dynamics')
    session = InspectorSession(config, stage)
    history = LossHistory(config, stage, logdir)
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = 'PixelModelInspector/1.0'

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, value):
            self._send(status, json.dumps(value).encode('utf-8'), 'application/json; charset=utf-8')

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/index.html'):
                self._send(200, page, 'text/html; charset=utf-8')
            elif path == '/api/state':
                try:
                    self._json(200, session.state())
                except Exception as exc:
                    self._json(500, {'error': str(exc)})
            elif path == '/api/loss':
                self._json(200, history.read())
            else:
                self._json(404, {'error': 'not found'})

        def do_POST(self):
            if urlsplit(self.path).path != '/api/action':
                self._json(404, {'error': 'not found'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size < 1 or size > MAX_REQUEST_BYTES:
                    raise ValueError('invalid request size')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('request body must be an object')
                self._json(200, session.act(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {'error': str(exc)})
            except Exception as exc:
                self._json(500, {'error': str(exc)})

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    url = f'http://{host}:{server.server_address[1]}/'
    print(f'Serving {stage} inspector at {url}', flush=True)
    print('Keep this process running; open the URL through an SSH port forward.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nInspector stopped.', flush=True)
    finally:
        server.server_close()
