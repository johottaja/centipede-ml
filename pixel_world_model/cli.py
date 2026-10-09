"""CLI entrypoints shared with the standalone Tkinter launcher."""
import argparse
import sys
import traceback
from pixel_world_model.config import DEFAULTS, SETTINGS, load, validate
from pixel_world_model.runtime import emit


def main(argv=None):
    parser = argparse.ArgumentParser(description='Two-stage pixel world model experiment')
    parser.add_argument('command', choices=('collect', 'train-vae', 'train-dynamics', 'inspect-vae', 'inspect-dynamics',
                                           'serve-vae', 'serve-dynamics'))
    parser.add_argument('--settings', default=str(SETTINGS))
    parser.add_argument('--host', default='127.0.0.1', help='Inspector bind address (default: loopback only)')
    parser.add_argument('--port', type=int, default=8765, help='Inspector HTTP port')
    parser.add_argument('--logdir', default=None, help='TensorBoard event directory for the custom loss chart')
    for key, value in DEFAULTS.items():
        parser.add_argument('--'+key.replace('_', '-'), type=type(value), default=None)
    args = parser.parse_args(argv)
    try:
        c = load(args.settings)
        c.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})
        validate(c)
        if args.command == 'collect':
            from pixel_world_model.data import collect
            collect(c)
        elif args.command.startswith('train-'):
            from pixel_world_model.train import train
            train(c, args.command.removeprefix('train-'))
        elif args.command.startswith('serve-'):
            import json
            import subprocess
            from pathlib import Path
            stage = args.command.removeprefix('serve-')
            if getattr(args, f'{stage}_checkpoint') is None:
                c[f'{stage}_checkpoint'] = str(Path(c[f'{stage}_output']) / 'current.pt')
            command = [sys.executable, '-m', 'streamlit', 'run',
                       str(Path(__file__).with_name('streamlit_inspector.py')),
                       '--server.address', args.host, '--server.port', str(args.port),
                       '--server.headless', 'true', '--', '--config', json.dumps(c), '--stage', stage]
            if args.logdir:
                command.extend(['--logdir', args.logdir])
            return subprocess.call(command)
        else:
            from pixel_world_model.visualize import visualize
            visualize(c, args.command.removeprefix('inspect-'))
    except Exception as exc:
        emit('error', text=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
