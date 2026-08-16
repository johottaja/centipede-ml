"""
Train a DQN agent on Centipede using Stable-Baselines3.

Observation: entity-centric float32 vector of length RELATIVE_OBS_SIZE (105).
Policy: MlpPolicy (two hidden layers).
The trained model is saved to models/dqn_centipede.zip.

Progress is emitted to stdout as newline-delimited JSON objects:
  {"type": "progress", "steps": N, "total": T, "pct": P, "elapsed": S, "eta": S, "steps_per_sec": F}
  {"type": "eval", "steps": N, "episodes": [...], "mean_score": S}
  {"type": "done", "elapsed": S}
  {"type": "error", "message": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.env import CentipedeEnv

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_centipede")


def _make_env_fn(
    seed: int = 0,
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


def _run_parallel_eval(
    model: DQN,
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
        _emit({
            "type": "eval",
            "steps": self.num_timesteps,
            "episodes": episodes,
            "mean_score": mean_score,
        })


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

    env = make_vec_env(
        n_envs=n_envs,
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
        reward_survival=reward_survival,
        reward_proximity_penalty=reward_proximity_penalty,
        proximity_distance_tiles=proximity_distance_tiles,
    )
    env_kwargs = dict(
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

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_centipede_ckpt",
        verbose=1,
    )
    progress_cb = ProgressCallback(total_timesteps)
    eval_cb = EvalProgressCallback(
        eval_freq=100_000,
        n_eval_episodes=10,
        env_kwargs=env_kwargs,
        seed=seed,
    )

    model = DQN(
        policy="MlpPolicy",
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
        policy_kwargs={"net_arch": net_arch},
        device=device,
        verbose=0,
        seed=seed,
    )

    _emit({"type": "start", "total": total_timesteps})
    t0 = time.monotonic()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, progress_cb, eval_cb],
        progress_bar=False,
    )

    model.save(MODEL_PATH)
    _emit({"type": "done", "elapsed": round(time.monotonic() - t0, 1)})
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
                        help="Comma-separated hidden layer sizes, e.g. 256,256")
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
