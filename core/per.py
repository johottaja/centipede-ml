"""Prioritized Experience Replay (Schaul et al., 2015).

Transitions are sampled with P(i) ∝ p_i^α, then the loss is scaled by
importance-sampling weights w_i = (N · P(i))^{-β} / max_j w_j so the
non-uniform sampling does not bias the gradient.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class _SegmentTree:
    """Binary segment tree over the next power-of-two capacity."""

    def __init__(self, size: int, identity: float, op: str) -> None:
        n = 1
        while n < size:
            n *= 2
        self._n = n
        self._op = op
        self.tree = np.full(2 * n, identity, dtype=np.float64)

    def __setitem__(self, idx: int, value: float) -> None:
        i = idx + self._n
        tree = self.tree
        tree[i] = value
        i //= 2
        if self._op == "sum":
            while i >= 1:
                tree[i] = tree[2 * i] + tree[2 * i + 1]
                i //= 2
        else:
            while i >= 1:
                tree[i] = min(tree[2 * i], tree[2 * i + 1])
                i //= 2

    def __getitem__(self, idx: int) -> float:
        return float(self.tree[idx + self._n])

    def total(self) -> float:
        return float(self.tree[1])

    def find_prefixsum(self, mass: float) -> int:
        i = 1
        n = self._n
        tree = self.tree
        while i < n:
            left = i * 2
            if tree[left] > mass:
                i = left
            else:
                mass -= tree[left]
                i = left + 1
        return i - n


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER on top of SB3's ReplayBuffer.

    New transitions are inserted at max priority so they are sampled at
    least once. `beta` is annealed externally (typically  β₀ → 1).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if optimize_memory_usage:
            raise ValueError(
                "PrioritizedReplayBuffer does not support optimize_memory_usage=True"
            )
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            **kwargs,
        )
        self.alpha = alpha
        self.eps = eps
        self.beta = 0.4
        n_slots = self.buffer_size * self.n_envs
        self.sum_tree = _SegmentTree(n_slots, identity=0.0, op="sum")
        self.min_tree = _SegmentTree(n_slots, identity=np.inf, op="min")
        self._max_priority = 1.0
        self.importance_weights: th.Tensor | None = None
        self.tree_indices: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        priority = self._max_priority ** self.alpha
        base = idx * self.n_envs
        for env_i in range(self.n_envs):
            self.sum_tree[base + env_i] = priority
            self.min_tree[base + env_i] = priority

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        n = self.size() * self.n_envs
        total = self.sum_tree.total()
        if n == 0 or total <= 0.0:
            raise ValueError("Cannot sample from an empty prioritized replay buffer")

        segment = total / batch_size
        p_min = self.min_tree.total()
        # (p_min / p_i)^β == w_i / max_j w_j  (global IS-weight normalisation)
        masses = (np.arange(batch_size) + np.random.random(batch_size)) * segment

        tree_indices = np.empty(batch_size, dtype=np.int64)
        batch_inds = np.empty(batch_size, dtype=np.int64)
        env_indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)

        last = n - 1
        for i, mass in enumerate(masses):
            idx = self.sum_tree.find_prefixsum(min(mass, total - 1e-12))
            if idx > last:
                idx = last
            p_i = self.sum_tree[idx]
            tree_indices[i] = idx
            batch_inds[i] = idx // self.n_envs
            env_indices[i] = idx % self.n_envs
            weights[i] = (p_min / p_i) ** self.beta if p_i > 0.0 else 0.0

        self.tree_indices = tree_indices
        self.importance_weights = self.to_torch(weights)
        return self._get_samples(batch_inds, env=env, env_indices=env_indices)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
        env_indices: Optional[np.ndarray] = None,
    ) -> ReplayBufferSamples:
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds))

        next_obs = self._normalize_obs(
            self.next_observations[batch_inds, env_indices, :], env
        )
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices]
             * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64))
        td_errors = np.nan_to_num(td_errors, nan=0.0, posinf=0.0)
        priorities = (td_errors + self.eps) ** self.alpha
        self._max_priority = max(self._max_priority, float(priorities.max()))
        for idx, priority in zip(indices, priorities):
            idx = int(idx)
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
