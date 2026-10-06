"""Experiment pixel capture, independent of root settings."""
from collections import deque
import torch
from torch.nn import functional as F
import numpy as np
from core.env import CentipedeEnv
from pixel_world_model.nets import FRAME_GAP, FRAME_STACK, IMG_SIZE


def rgb_to_gray84(rgb, resolution=IMG_SIZE):
    gray = np.rint(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    resized = F.interpolate(torch.from_numpy(gray)[None, None], (resolution, resolution), mode="area")
    return resized[0, 0].numpy().round().clip(0, 255).astype(np.uint8)


def capture_frame(env, resolution=IMG_SIZE):
    return rgb_to_gray84(env.render(), resolution)


def make_rollout_env(seed=None):
    env = CentipedeEnv(render_mode='rgb_array', frame_skip=FRAME_GAP)
    return env


class FrameStacker:
    def __init__(self, stack_size=FRAME_STACK):
        self._frames = deque(maxlen=stack_size)

    def push(self, frame):
        self._frames.append(frame)

    def clear(self):
        self._frames.clear()

    def ready(self):
        return len(self._frames) == self._frames.maxlen

    def as_array(self):
        return np.stack(self._frames).astype(np.uint8)
