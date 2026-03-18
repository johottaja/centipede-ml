"""
Train a DQN agent on Centipede using Stable-Baselines3.

Observations are preprocessed: grayscale → 84×84 → 4-frame stack (channels-first).
The trained model is saved to models/dqn_centipede.zip.

Usage:
    uv run python train.py [--timesteps N] [--seed S]
"""
from __future__ import annotations

import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from env import CentipedeEnv

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_centipede")

OBS_SIZE = 84
N_STACK = 4


# ---------------------------------------------------------------------------
# Preprocessing wrapper: RGB (H,W,3) → grayscale (1, 84, 84) uint8
# ---------------------------------------------------------------------------

class PreprocessObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, OBS_SIZE, OBS_SIZE), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)          # (H, W)
        small = cv2.resize(gray, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)
        return small[np.newaxis, :, :]                         # (1, 84, 84)


# ---------------------------------------------------------------------------
# After FrameStackObservation the shape is (N_STACK, 1, 84, 84).
# Squeeze the channel dim → (N_STACK, 84, 84) which SB3 CnnPolicy expects.
# ---------------------------------------------------------------------------

class SqueezeStack(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old = env.observation_space
        # old.shape = (N_STACK, 1, H, W)
        n, _, h, w = old.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n, h, w), dtype=np.uint8
        )

    def observation(self, obs) -> np.ndarray:
        arr = np.asarray(obs)          # (N_STACK, 1, H, W)
        return arr[:, 0, :, :]         # (N_STACK, H, W)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(seed: int = 0) -> gym.Env:
    env = CentipedeEnv(render_mode=None)
    env = Monitor(env)
    env = PreprocessObs(env)                          # (1, 84, 84)
    env = FrameStackObservation(env, stack_size=N_STACK)  # (4, 1, 84, 84)
    env = SqueezeStack(env)                           # (4, 84, 84)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_freq: int = 10_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            pct = 100 * self.num_timesteps / self.total_timesteps
            print(f"  [{pct:5.1f}%] steps={self.num_timesteps:,}", flush=True)
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 1_000_000, seed: int = 0) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    env = make_env(seed=seed)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_centipede_ckpt",
        verbose=1,
    )
    progress_cb = ProgressCallback(total_timesteps)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=50_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        device=device,
        verbose=0,
        seed=seed,
    )

    print(f"Training DQN for {total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, progress_cb],
        progress_bar=False,
    )

    model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}.zip")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, seed=args.seed)
