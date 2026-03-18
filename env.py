"""
Gymnasium environment for Centipede.

Observation : RGB pixel array  (HEIGHT, WIDTH, 3)  uint8
Action space: Discrete(10)  – see game.ACTION_* constants
Reward       : score delta per step
Terminated   : player loses all lives
"""
from __future__ import annotations

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from game import (
    GameEngine,
    WIDTH, HEIGHT, NUM_ACTIONS,
)


class CentipedeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            f"Unsupported render_mode: {render_mode}"
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._engine = GameEngine()
        self._surf: pygame.Surface | None = None   # off-screen surface
        self._window: pygame.Surface | None = None  # on-screen window (human mode)
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

    # ------------------------------------------------------------------
    def _init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 16)
        if self._surf is None:
            self._surf = pygame.Surface((WIDTH, HEIGHT))
        if self.render_mode == "human" and self._window is None:
            self._window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Centipede")
            self._clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._engine.reset(seed=seed)
        self._init_pygame()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, terminated, truncated = self._engine.step(int(action))
        obs = self._get_obs()
        info = {"score": self._engine.score, "lives": self._engine.player.lives}
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        self._init_pygame()
        self._engine.render(self._surf, self._font)

        if self.render_mode == "human":
            self._window.blit(self._surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self._surf), axes=(1, 0, 2)
            )
        return None

    # ------------------------------------------------------------------
    def close(self):
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        if pygame.get_init():
            pygame.quit()

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        self._engine.render(self._surf)
        return np.transpose(
            pygame.surfarray.array3d(self._surf), axes=(1, 0, 2)
        )
