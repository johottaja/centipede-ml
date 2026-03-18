# Centipede

A Centipede arcade game with a Gymnasium environment and a Tkinter GUI for training and watching RL agents.

## Requirements

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Running

### GUI (recommended)

Launches a three-tab window — Play, Train Agent, Watch Agent.

```bash
uv run python main.py
```

### Play yourself

```bash
uv run python -m core.play
```

Arrow keys or WASD to move, Space to fire. Press `R` to restart after game over.

### Train an agent

```bash
uv run python -m core.train
```

The final model is saved to `models/dqn_centipede.zip`. Checkpoints are saved to `models/dqn_centipede_ckpt_<N>_steps.zip` every 100k steps. After each checkpoint, 10 evaluation games are run and logged to stdout as JSON.

Common options:

```
--timesteps INT        Total training steps (default: 1_000_000)
--seed INT             Random seed (default: 0)
--learning-rate FLOAT  (default: 0.0001)
--net-arch STR         Hidden layer sizes, e.g. 256,256 (default: 256,256)
```

Run `uv run python -m core.train --help` for the full list.

### Watch a trained agent

```bash
uv run python -m core.watch
```

Options:

```
--model PATH      Path to model file without .zip extension (default: models/dqn_centipede)
--episodes INT    Number of episodes to watch (default: 5)
--fps INT         Playback speed (default: 60)
```
