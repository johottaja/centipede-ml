"""
Train a C51 (Categorical DQN) agent on Centipede using Stable-Baselines3.

Observation: uint8 occupancy grid (31, 30, 24) — 4 stacked frames × 6 channels.
Policy: C51Policy with custom GridCNN feature extractor + distributional MLP head.
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
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.c51 import C51, C51Policy
from core.env import CentipedeEnv
from core.hparams import SETTINGS_PATH, load_settings_defaults
from core.networks import GridCNN

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
    model: C51,
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
    n_steps: int = 4,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1_000,
    exploration_fraction: float = 0.1,
    exploration_final_eps: float = 0.01,
    net_arch: list[int] | None = None,
    eval_freq: int = 30_000,
    checkpoint_freq: int = 100_000,
    n_atoms: int = 51,
    v_min: float = -10_000.0,
    v_max: float = 10_000.0,
    prioritized_replay: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
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
    if isinstance(prioritized_replay, str):
        prioritized_replay = prioritized_replay.lower() in ("1", "true", "yes")

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

    # save_freq is per vec-env step; divide by n_envs to hit every checkpoint_freq timesteps
    checkpoint_cb = LoggingCheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
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

    model = C51(
        policy=C51Policy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        n_steps=n_steps,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs={
            "features_extractor_class": GridCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": net_arch,
            "normalize_images": True,
            "n_atoms": n_atoms,
            "v_min": v_min,
            "v_max": v_max,
        },
        prioritized_replay=prioritized_replay,
        prioritized_replay_alpha=per_alpha,
        prioritized_replay_beta=per_beta,
        device=device,
        verbose=0,
        seed=seed,
    )

    _emit({"type": "start", "total": total_timesteps})
    _log(
        f"training (C51) | {total_timesteps:,} steps | {n_envs} envs | "
        f"{n_atoms} atoms [{v_min:,.0f}, {v_max:,.0f}] | "
        f"n-step={n_steps} | "
        f"{'PER α=' + str(per_alpha) + ' β=' + str(per_beta) if prioritized_replay else 'uniform replay'} | "
        f"eval every {eval_freq:,} steps | checkpoint every {checkpoint_freq:,} steps | seed {seed}"
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--settings",
        default=SETTINGS_PATH,
        help=f"Path to settings JSON (default: {SETTINGS_PATH})",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    settings_defaults = load_settings_defaults(pre_args.settings)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Train a C51 agent. Loads hyperparameters from settings.json; "
                    "CLI flags override.",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--n-steps", type=int,
                        help="N-step returns (Rainbow). 1 = one-step TD")
    parser.add_argument("--train-freq", type=int)
    parser.add_argument("--gradient-steps", type=int)
    parser.add_argument("--target-update-interval", type=int)
    parser.add_argument("--exploration-fraction", type=float)
    parser.add_argument("--exploration-final-eps", type=float)
    parser.add_argument("--net-arch", type=str,
                        help="Comma-separated MLP head layer sizes after CNN, e.g. 256,256")
    parser.add_argument("--eval-freq", type=int,
                        help="Run eval games every N training steps (default: 30000)")
    parser.add_argument("--checkpoint-freq", type=int,
                        help="Save a model checkpoint every N training steps (default: 100000)")
    parser.add_argument("--n-atoms", type=int,
                        help="Number of atoms for C51 return distribution (default: 51)")
    parser.add_argument("--v-min", type=float,
                        help="Minimum support value for C51 atoms")
    parser.add_argument("--v-max", type=float,
                        help="Maximum support value for C51 atoms")
    parser.add_argument("--prioritized-replay", type=str, choices=["true", "false"],
                        help="Use prioritized experience replay (Schaul et al. 2015)")
    parser.add_argument("--per-alpha", type=float,
                        help="PER priority exponent (0 = uniform)")
    parser.add_argument("--per-beta", type=float,
                        help="PER importance-sampling exponent at start of training (annealed to 1)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress human-readable logs on stderr (JSON stdout only)")
    parser.add_argument("--reward-mushroom-hit", type=int)
    parser.add_argument("--reward-mushroom-destroy", type=int)
    parser.add_argument("--reward-body-hit", type=int)
    parser.add_argument("--reward-head-hit", type=int)
    parser.add_argument("--reward-depth-discount", type=float)
    parser.add_argument("--reward-depth-discount-fn", type=str,
                        choices=["linear", "exponential"])
    parser.add_argument("--reward-spider-hit", type=int)
    parser.add_argument("--reward-spider-penalty", type=int)
    parser.add_argument("--reward-centipede-penalty", type=int)
    parser.add_argument("--reward-survival", type=float)
    parser.add_argument("--reward-proximity-penalty", type=float)
    parser.add_argument("--proximity-distance-tiles", type=int)
    parser.set_defaults(**settings_defaults)
    args = parser.parse_args(remaining)
    if not args.quiet:
        _log(f"loaded settings from {pre_args.settings}")
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
        n_steps=args.n_steps,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        net_arch=[int(x) for x in args.net_arch.split(",")],
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        prioritized_replay=args.prioritized_replay,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
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
