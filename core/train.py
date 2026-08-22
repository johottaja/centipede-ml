"""
Train a Double DQN agent on Centipede using Stable-Baselines3.

Observation: uint8 occupancy grid (31, 30, 20) — 4 stacked frames × 5 channels.
Policy: CnnPolicy with a small custom CNN + MLP head.
The trained model is saved to models/dqn_centipede.zip.

Progress is emitted to stdout as newline-delimited JSON (for the GUI progress window).
Human-readable status lines are written to stderr when not using --quiet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.env import CentipedeEnv
from core.game import ROWS, COLS

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_centipede")
FRAME_SKIP = 4


def list_saved_models() -> list[tuple[str, str]]:
    """Return (path_without_zip, label) pairs, newest file first."""
    if not os.path.isdir(MODEL_DIR):
        return []

    entries: list[tuple[str, str, float]] = []

    final_zip = MODEL_PATH + ".zip"
    if os.path.isfile(final_zip):
        entries.append((MODEL_PATH, "Final model", os.path.getmtime(final_zip)))

    prefix = "dqn_centipede_ckpt_"
    suffix = "_steps.zip"
    for fname in os.listdir(MODEL_DIR):
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue
        full = os.path.join(MODEL_DIR, fname)
        path = full[:-4]
        steps = fname[len(prefix):-len(suffix)]
        try:
            steps_fmt = f"{int(steps):,}"
        except ValueError:
            steps_fmt = steps
        entries.append((
            path,
            f"Checkpoint — {steps_fmt} steps",
            os.path.getmtime(full),
        ))

    entries.sort(key=lambda e: e[2], reverse=True)
    return [(path, label) for path, label, _ in entries]


class GridCNN(BaseFeaturesExtractor):
    """Small CNN for 31×30 occupancy grids (channels-first input from SB3)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(1, n_input_channels, ROWS, COLS)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class DoubleDQN(DQN):
    """DQN with Double-Q target: online net selects, target net evaluates."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions,
                )
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def _make_env_fn(
    seed: int = 0,
    frame_skip: int = FRAME_SKIP,
    reward_mushroom_hit: int = 1,
    reward_mushroom_destroy: int = 5,
    reward_body_hit: int = 10,
    reward_head_hit: int = 100,
    reward_depth_discount: float = 0.0,
    reward_depth_discount_fn: str = "linear",
    reward_spider_hit: int = 300,
    reward_spider_penalty: int = 1000,
    reward_centipede_penalty: int = 1000,
    reward_survival: float = 0.01,
    reward_proximity_penalty: float = 1.0,
    proximity_distance_tiles: int = 3,
):
    """Return a thunk that creates and seeds one monitored env (for VecEnv factories)."""
    def _thunk():
        env = CentipedeEnv(
            render_mode=None,
            frame_skip=frame_skip,
            reward_mushroom_hit=reward_mushroom_hit,
            reward_mushroom_destroy=reward_mushroom_destroy,
            reward_body_hit=reward_body_hit,
            reward_head_hit=reward_head_hit,
            reward_depth_discount=reward_depth_discount,
            reward_depth_discount_fn=reward_depth_discount_fn,
            reward_spider_hit=reward_spider_hit,
            reward_spider_penalty=reward_spider_penalty,
            reward_centipede_penalty=reward_centipede_penalty,
            reward_survival=reward_survival,
            reward_proximity_penalty=reward_proximity_penalty,
            proximity_distance_tiles=proximity_distance_tiles,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def make_vec_env(
    n_envs: int = 1,
    seed: int = 0,
    frame_skip: int = FRAME_SKIP,
    reward_mushroom_hit: int = 1,
    reward_mushroom_destroy: int = 5,
    reward_body_hit: int = 10,
    reward_head_hit: int = 100,
    reward_depth_discount: float = 0.0,
    reward_depth_discount_fn: str = "linear",
    reward_spider_hit: int = 300,
    reward_spider_penalty: int = 1000,
    reward_centipede_penalty: int = 1000,
    reward_survival: float = 0.01,
    reward_proximity_penalty: float = 1.0,
    proximity_distance_tiles: int = 3,
):
    kwargs = dict(
        frame_skip=frame_skip,
        reward_mushroom_hit=reward_mushroom_hit,
        reward_mushroom_destroy=reward_mushroom_destroy,
        reward_body_hit=reward_body_hit,
        reward_head_hit=reward_head_hit,
        reward_depth_discount=reward_depth_discount,
        reward_depth_discount_fn=reward_depth_discount_fn,
        reward_spider_hit=reward_spider_hit,
        reward_spider_penalty=reward_spider_penalty,
        reward_centipede_penalty=reward_centipede_penalty,
        reward_survival=reward_survival,
        reward_proximity_penalty=reward_proximity_penalty,
        proximity_distance_tiles=proximity_distance_tiles,
    )
    fns = [_make_env_fn(seed=seed + i, **kwargs) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="fork")


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


_quiet = False


def _log(msg: str) -> None:
    if not _quiet:
        print(msg, file=sys.stderr, flush=True)


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _log_eval(steps: int, episodes: list[dict]) -> None:
    if not episodes:
        _log(f"eval @ {steps:,} steps — no episodes recorded")
        return
    scores = [ep["score"] for ep in episodes]
    segs = [ep["segments_destroyed"] for ep in episodes]
    spiders = [ep["spiders_destroyed"] for ep in episodes]
    mean_score = sum(scores) / len(scores)
    mean_segs = sum(segs) / len(segs)
    mean_spiders = sum(spiders) / len(spiders)
    _log(
        f"eval @ {steps:,} steps | "
        f"mean score {mean_score:,.1f} | "
        f"avg segments {mean_segs:.1f} | "
        f"avg spiders {mean_spiders:.1f} | "
        f"min/max score {min(scores):,.1f}/{max(scores):,.1f}"
    )
    for i, ep in enumerate(episodes, 1):
        _log(
            f"  game {i:2d}: score {ep['score']:,.1f}  "
            f"segments {ep['segments_destroyed']}  "
            f"spiders {ep['spiders_destroyed']}"
        )


def _run_parallel_eval(
    model: DoubleDQN,
    n_eval_episodes: int,
    env_kwargs: dict,
    seed: int,
    timesteps: int,
) -> list[dict]:
    """Run eval episodes in parallel via SubprocVecEnv (one env per episode)."""
    eval_env = make_vec_env(
        n_envs=n_eval_episodes,
        seed=seed + timesteps,
        **env_kwargs,
    )
    episodes: list[dict] = []

    def _callback(locals_: dict, globals_: dict) -> None:
        if not locals_["done"]:
            return
        info = locals_["info"]
        if "episode" not in info:
            return
        episodes.append({
            "score": round(float(info.get("score", info["episode"]["r"])), 1),
            "segments_destroyed": int(info.get("segments_destroyed", 0)),
            "spiders_destroyed": int(info.get("spiders_destroyed", 0)),
        })

    try:
        evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            callback=_callback,
            warn=False,
        )
    finally:
        eval_env.close()

    return episodes


class LoggingCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that logs saves to stderr."""

    def _on_step(self) -> bool:
        save_now = self.save_freq > 0 and self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if save_now:
            path = self._checkpoint_path(extension="zip")
            _log(f"checkpoint saved @ {self.model.num_timesteps:,} steps → {path}")
        return result


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
            _log(
                f"progress | {self.num_timesteps:,} / {self.total_timesteps:,} "
                f"({pct * 100:.1f}%) | {sps:,.0f} steps/s | "
                f"elapsed {_fmt_duration(elapsed)} | eta {_fmt_duration(eta)}"
            )
        return True


class EvalProgressCallback(BaseCallback):
    """Run deterministic eval episodes in parallel and emit results to stdout."""

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int,
        env_kwargs: dict,
        seed: int = 0,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.env_kwargs = env_kwargs
        self.seed = seed
        self._next_eval = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_freq
        return True

    def _run_eval(self) -> None:
        _log(f"running eval ({self.n_eval_episodes} games) @ {self.num_timesteps:,} steps…")
        episodes = _run_parallel_eval(
            self.model,
            self.n_eval_episodes,
            self.env_kwargs,
            self.seed,
            self.num_timesteps,
        )
        mean_score = round(
            sum(ep["score"] for ep in episodes) / max(1, len(episodes)), 1
        )
        mean_segments = round(
            sum(ep["segments_destroyed"] for ep in episodes) / max(1, len(episodes)), 1
        )
        mean_spiders = round(
            sum(ep["spiders_destroyed"] for ep in episodes) / max(1, len(episodes)), 1
        )
        _emit({
            "type": "eval",
            "steps": self.num_timesteps,
            "episodes": episodes,
            "mean_score": mean_score,
            "mean_segments_destroyed": mean_segments,
            "mean_spiders_destroyed": mean_spiders,
        })
        _log_eval(self.num_timesteps, episodes)


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    seed: int = 0,
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    batch_size: int = 64,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1_000,
    exploration_fraction: float = 0.1,
    exploration_final_eps: float = 0.01,
    net_arch: list[int] | None = None,
    eval_freq: int = 30_000,
    reward_mushroom_hit: int = 1,
    reward_mushroom_destroy: int = 5,
    reward_body_hit: int = 10,
    reward_head_hit: int = 100,
    reward_depth_discount: float = 0.0,
    reward_depth_discount_fn: str = "linear",
    reward_spider_hit: int = 300,
    reward_spider_penalty: int = 1000,
    reward_centipede_penalty: int = 1000,
    reward_survival: float = 0.01,
    reward_proximity_penalty: float = 1.0,
    proximity_distance_tiles: int = 3,
    quiet: bool = False,
) -> None:
    global _quiet
    _quiet = quiet

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
    _log(f"device: {device}")

    env_kwargs: dict[str, Any] = dict(
        frame_skip=FRAME_SKIP,
        reward_mushroom_hit=reward_mushroom_hit,
        reward_mushroom_destroy=reward_mushroom_destroy,
        reward_body_hit=reward_body_hit,
        reward_head_hit=reward_head_hit,
        reward_depth_discount=reward_depth_discount,
        reward_depth_discount_fn=reward_depth_discount_fn,
        reward_spider_hit=reward_spider_hit,
        reward_spider_penalty=reward_spider_penalty,
        reward_centipede_penalty=reward_centipede_penalty,
        reward_survival=reward_survival,
        reward_proximity_penalty=reward_proximity_penalty,
        proximity_distance_tiles=proximity_distance_tiles,
    )

    env = make_vec_env(n_envs=n_envs, seed=seed, **env_kwargs)

    checkpoint_cb = LoggingCheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_centipede_ckpt",
        verbose=0,
    )
    progress_cb = ProgressCallback(total_timesteps)
    eval_cb = EvalProgressCallback(
        eval_freq=eval_freq,
        n_eval_episodes=10,
        env_kwargs=env_kwargs,
        seed=seed,
    )

    model = DoubleDQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs={
            "features_extractor_class": GridCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": net_arch,
            "normalize_images": True,
        },
        device=device,
        verbose=0,
        seed=seed,
    )

    _emit({"type": "start", "total": total_timesteps})
    _log(
        f"training | {total_timesteps:,} steps | {n_envs} envs | "
        f"eval every {eval_freq:,} steps | seed {seed}"
    )
    t0 = time.monotonic()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, progress_cb, eval_cb],
        progress_bar=False,
    )

    model.save(MODEL_PATH)
    elapsed = round(time.monotonic() - t0, 1)
    _emit({"type": "done", "elapsed": elapsed})
    _log(f"done in {_fmt_duration(elapsed)} | saved {MODEL_PATH}.zip")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--net-arch", type=str, default="256,256",
                        help="Comma-separated MLP head layer sizes after CNN, e.g. 256,256")
    parser.add_argument("--eval-freq", type=int, default=30_000,
                        help="Run eval games every N training steps (default: 30000)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress human-readable logs on stderr (JSON stdout only)")
    parser.add_argument("--reward-mushroom-hit", type=int, default=1)
    parser.add_argument("--reward-mushroom-destroy", type=int, default=5)
    parser.add_argument("--reward-body-hit", type=int, default=10)
    parser.add_argument("--reward-head-hit", type=int, default=100)
    parser.add_argument("--reward-depth-discount", type=float, default=0.0)
    parser.add_argument("--reward-depth-discount-fn", type=str, default="linear",
                        choices=["linear", "exponential"])
    parser.add_argument("--reward-spider-hit", type=int, default=300)
    parser.add_argument("--reward-spider-penalty", type=int, default=1000)
    parser.add_argument("--reward-centipede-penalty", type=int, default=1000)
    parser.add_argument("--reward-survival", type=float, default=0.01)
    parser.add_argument("--reward-proximity-penalty", type=float, default=1.0)
    parser.add_argument("--proximity-distance-tiles", type=int, default=3)
    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        net_arch=[int(x) for x in args.net_arch.split(",")],
        eval_freq=args.eval_freq,
        quiet=args.quiet,
        reward_mushroom_hit=args.reward_mushroom_hit,
        reward_mushroom_destroy=args.reward_mushroom_destroy,
        reward_body_hit=args.reward_body_hit,
        reward_head_hit=args.reward_head_hit,
        reward_depth_discount=args.reward_depth_discount,
        reward_depth_discount_fn=args.reward_depth_discount_fn,
        reward_spider_hit=args.reward_spider_hit,
        reward_spider_penalty=args.reward_spider_penalty,
        reward_centipede_penalty=args.reward_centipede_penalty,
        reward_survival=args.reward_survival,
        reward_proximity_penalty=args.reward_proximity_penalty,
        proximity_distance_tiles=args.proximity_distance_tiles,
    )
