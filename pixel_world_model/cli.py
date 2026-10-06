"""CLI entrypoints shared with the standalone Tkinter launcher."""
import argparse
import sys
import traceback
from pixel_world_model.config import DEFAULTS, SETTINGS, load, validate
from pixel_world_model.runtime import emit


def main(argv=None):
    parser = argparse.ArgumentParser(description='Two-stage pixel world model experiment')
    parser.add_argument('command', choices=('collect', 'train-vae', 'train-dynamics', 'inspect-vae', 'inspect-dynamics'))
    parser.add_argument('--settings', default=str(SETTINGS))
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
