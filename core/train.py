"""
Train a PPO agent on Centipede using Stable-Baselines3.

Observation: flat grid vector of length COLS*ROWS (930 ints, values 0-5).
Policy: MlpPolicy (two hidden layers).
The trained model is saved to models/ppo_centipede.zip.

Progress is emitted to stdout as newline-delimited JSON objects:
  {"type": "progress", "steps": N, "total": T, "pct": P, "elapsed": S, "eta": S, "steps_per_sec": F}
  {"type": "done", "elapsed": S}
  {"type": "error", "message": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from core.env import CentipedeEnv

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_centipede")


def make_env(
    seed: int = 0,
    reward_mushroom_hit: int = 1,
    reward_mushroom_destroy: int = 5,
    reward_body_hit: int = 10,
    reward_head_hit: int = 100,
    reward_depth_discount: float = 0.0,
    reward_depth_discount_fn: str = "linear",
    reward_spider_hit: int = 300,
    reward_spider_penalty: int = 0,
    reward_centipede_penalty: int = 0,
) -> CentipedeEnv:
    env = CentipedeEnv(
        render_mode=None,
        reward_mushroom_hit=reward_mushroom_hit,
        reward_mushroom_destroy=reward_mushroom_destroy,
        reward_body_hit=reward_body_hit,
        reward_head_hit=reward_head_hit,
        reward_depth_discount=reward_depth_discount,
        reward_depth_discount_fn=reward_depth_discount_fn,
        reward_spider_hit=reward_spider_hit,
        reward_spider_penalty=reward_spider_penalty,
        reward_centipede_penalty=reward_centipede_penalty,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, emit_freq: int = 5_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.emit_freq = emit_freq
        self._last_emit = 0
        self._t0 = 0.0

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_emit >= self.emit_freq:
            self._last_emit = self.num_timesteps
            elapsed = time.monotonic() - self._t0
            pct = self.num_timesteps / self.total_timesteps
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            remaining = self.total_timesteps - self.num_timesteps
            eta = remaining / sps if sps > 0 else 0.0
            _emit({
                "type": "progress",
                "steps": self.num_timesteps,
                "total": self.total_timesteps,
                "pct": round(pct * 100, 2),
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
                "steps_per_sec": round(sps, 1),
            })
        return True


def train(
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    net_arch: list[int] | None = None,
    reward_mushroom_hit: int = 1,
    reward_mushroom_destroy: int = 5,
    reward_body_hit: int = 10,
    reward_head_hit: int = 100,
    reward_depth_discount: float = 0.0,
    reward_depth_discount_fn: str = "linear",
    reward_spider_hit: int = 300,
    reward_spider_penalty: int = 0,
    reward_centipede_penalty: int = 0,
) -> None:
    if net_arch is None:
        net_arch = [256, 256]

    os.makedirs(MODEL_DIR, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    _emit({"type": "device", "device": device})

    env = make_env(
        seed=seed,
        reward_mushroom_hit=reward_mushroom_hit,
        reward_mushroom_destroy=reward_mushroom_destroy,
        reward_body_hit=reward_body_hit,
        reward_head_hit=reward_head_hit,
        reward_depth_discount=reward_depth_discount,
        reward_depth_discount_fn=reward_depth_discount_fn,
        reward_spider_hit=reward_spider_hit,
        reward_spider_penalty=reward_spider_penalty,
        reward_centipede_penalty=reward_centipede_penalty,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_centipede_ckpt",
        verbose=1,
    )
    progress_cb = ProgressCallback(total_timesteps)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs={"net_arch": net_arch},
        device=device,
        verbose=0,
        seed=seed,
    )

    _emit({"type": "start", "total": total_timesteps})
    t0 = time.monotonic()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, progress_cb],
        progress_bar=False,
    )

    model.save(MODEL_PATH)
    _emit({"type": "done", "elapsed": round(time.monotonic() - t0, 1)})
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=str, default="256,256",
                        help="Comma-separated hidden layer sizes, e.g. 256,256")
    parser.add_argument("--reward-mushroom-hit", type=int, default=1)
    parser.add_argument("--reward-mushroom-destroy", type=int, default=5)
    parser.add_argument("--reward-body-hit", type=int, default=10)
    parser.add_argument("--reward-head-hit", type=int, default=100)
    parser.add_argument("--reward-depth-discount", type=float, default=0.0)
    parser.add_argument("--reward-depth-discount-fn", type=str, default="linear",
                        choices=["linear", "exponential"])
    parser.add_argument("--reward-spider-hit", type=int, default=300)
    parser.add_argument("--reward-spider-penalty", type=int, default=0)
    parser.add_argument("--reward-centipede-penalty", type=int, default=0)
    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=[int(x) for x in args.net_arch.split(",")],
        reward_mushroom_hit=args.reward_mushroom_hit,
        reward_mushroom_destroy=args.reward_mushroom_destroy,
        reward_body_hit=args.reward_body_hit,
        reward_head_hit=args.reward_head_hit,
        reward_depth_discount=args.reward_depth_discount,
        reward_depth_discount_fn=args.reward_depth_discount_fn,
        reward_spider_hit=args.reward_spider_hit,
        reward_spider_penalty=args.reward_spider_penalty,
        reward_centipede_penalty=args.reward_centipede_penalty,
    )
