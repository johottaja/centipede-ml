"""
C51 (Categorical DQN) — distributional RL with a fixed-support return distribution.

Each Q(s, a) is represented as a categorical distribution over `n_atoms` values
evenly spaced in [v_min, v_max]. Training minimises cross-entropy against the
projected Bellman target distribution (Bellemare et al., 2017).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.dqn import DQN


def project_distribution(
    next_dist: th.Tensor,
    rewards: th.Tensor,
    dones: th.Tensor,
    gamma: float,
    support: th.Tensor,
) -> th.Tensor:
    """Project the bootstrap distribution onto the fixed atom support."""
    batch = next_dist.shape[0]
    n_atoms = support.shape[0]
    v_min = support[0]
    v_max = support[-1]
    delta_z = (v_max - v_min) / (n_atoms - 1)

    rewards = rewards.squeeze(-1)
    dones = dones.squeeze(-1).float()

    tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l_eq_u = l == u
    u = th.where(l_eq_u, l + 1, u)
    l = th.where(l_eq_u, l - 1, l)

    l = l.clamp(0, n_atoms - 1)
    u = u.clamp(0, n_atoms - 1)

    offset = (th.arange(batch, device=next_dist.device) * n_atoms).unsqueeze(1)
    proj = th.zeros(batch, n_atoms, device=next_dist.device)

    ml = (next_dist * (u.float() - b)).view(-1)
    mu = (next_dist * (b - l.float())).view(-1)
    proj.view(-1).index_add_(0, (l + offset).view(-1), ml)
    proj.view(-1).index_add_(0, (u + offset).view(-1), mu)
    return proj


class C51QNetwork(BaseModel):
    """Q-network that outputs a categorical distribution per action."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        n_atoms: int = 51,
        v_min: float = -10_000.0,
        v_max: float = 10_000.0,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64, 64]
        self.n_atoms = n_atoms
        self.register_buffer("support", th.linspace(v_min, v_max, n_atoms))
        action_dim = int(self.action_space.n)
        mlp = create_mlp(features_dim, action_dim * n_atoms, net_arch, activation_fn)
        self.logits_net = nn.Sequential(*mlp)

    def _logits(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        batch = features.shape[0]
        action_dim = int(self.action_space.n)
        return self.logits_net(features).view(batch, action_dim, self.n_atoms)

    def dist(self, obs: PyTorchObs) -> th.Tensor:
        return F.softmax(self._logits(obs), dim=-1)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        probs = self.dist(obs)
        return (probs * self.support).sum(dim=-1)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self(observation).argmax(dim=1).reshape(-1)


class C51Policy(DQNPolicy):
    """DQN policy with C51QNetwork heads."""

    def __init__(
        self,
        *args: Any,
        n_atoms: int = 51,
        v_min: float = -10_000.0,
        v_max: float = 10_000.0,
        **kwargs: Any,
    ):
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> C51QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return C51QNetwork(
            **net_args,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
        ).to(self.device)


class C51(DQN):
    """Categorical DQN with double-Q action selection for the target distribution."""

    policy: C51Policy

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        support = self.q_net.support

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                next_dist_online = self.q_net.dist(replay_data.next_observations)
                next_q = (next_dist_online * support).sum(dim=-1)
                next_actions = next_q.argmax(dim=1)

                next_dist_target = self.q_net_target.dist(replay_data.next_observations)
                batch_idx = th.arange(next_actions.shape[0], device=next_actions.device)
                next_dist = next_dist_target[batch_idx, next_actions]

                proj = project_distribution(
                    next_dist,
                    replay_data.rewards,
                    replay_data.dones,
                    float(discounts) if isinstance(discounts, (int, float)) else discounts,
                    support,
                )

            logits = self.q_net._logits(replay_data.observations)
            actions = replay_data.actions.long().squeeze(-1)
            batch_idx = th.arange(actions.shape[0], device=actions.device)
            action_logits = logits[batch_idx, actions]
            log_probs = F.log_softmax(action_logits, dim=-1)
            loss = -(proj * log_probs).sum(dim=-1).mean()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
