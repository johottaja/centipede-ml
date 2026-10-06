"""Load a trained C51 Centipede agent for world-model rollouts."""
from __future__ import annotations

import os

import numpy as np
import torch as th

from core.c51 import C51
from pathlib import Path

MODEL_PATH = "models/dqn_centipede"


def resolve_policy_model(path: str | None) -> str:
    """Path without ``.zip``; default to newest saved C51 under ``models/``."""
    if path:
        path = path.removesuffix(".zip")
        if not os.path.isfile(path + ".zip"):
            raise FileNotFoundError(f"No model at {path}.zip")
        return path

    if os.path.isfile(MODEL_PATH + ".zip"):
        return MODEL_PATH

    saved = sorted(Path("models").glob("dqn_centipede_ckpt_*_steps.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not saved:
        raise FileNotFoundError(
            "No C51 model found under models/. Train with core.train or pass --policy-model."
        )
    return str(saved[0].with_suffix(""))


def load_c51_policy(model_path: str, device: th.device) -> C51:
    model = C51.load(model_path, device=device)
    model.policy.set_training_mode(False)
    return model


def greedy_action(model: C51, obs: np.ndarray) -> int:
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        q = model.q_net(obs_t)
        return int(q.argmax(dim=1).item())
