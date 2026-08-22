"""
Gymnasium environment for Centipede.

Observation : uint8 occupancy grid, shape (ROWS, COLS, OCCUPANCY_CHANNELS * frame_skip).
              Each frame contributes 6 channels (player, mushrooms, heads, body,
              spiders, bullet) with values 0–255 encoding fractional tile occupancy.
              Default frame_skip=4 stacks 4 consecutive frames → shape (31, 30, 24).
Action space: Discrete(9) – 5 moves + 4 move-and-fire (no standing fire)
Reward       : score delta per agent step (summed across repeated frames)
Terminated   : player loses all lives
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from core.game import (
    GameEngine,
    ROWS,
    COLS,
    WIDTH,
    HEIGHT,
    NUM_ACTIONS,
)


class CentipedeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 4,
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
        window_size: tuple[int, int] | None = None,
        present_fn: Callable[[pygame.Surface], None] | None = None,
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            f"Unsupported render_mode: {render_mode}"
        assert frame_skip >= 1
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.render_fps = self.metadata["render_fps"]
        self.window_size = window_size or (WIDTH, HEIGHT)
        self.present_fn = present_fn

        n_channels = GameEngine.OCCUPANCY_CHANNELS * frame_skip
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(ROWS, COLS, n_channels),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._engine = GameEngine(
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
        self._frame_buf = np.zeros(
            (GameEngine.OCCUPANCY_CHANNELS, ROWS, COLS), dtype=np.float32
        )
        self._obs_buf = np.zeros((ROWS, COLS, n_channels), dtype=np.uint8)
        self._surf: pygame.Surface | None = None
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

    # ------------------------------------------------------------------
    def _init_pygame(self):
        """Initialise pygame and allocate surfaces. No-op when headless."""
        if self.render_mode is None:
            return
        if not pygame.get_init():
            pygame.init()
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 16)
        if self._surf is None:
            self._surf = pygame.Surface((WIDTH, HEIGHT))
        if self.render_mode == "human" and self._window is None:
            self._window = pygame.display.set_mode(self.window_size)
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
        if self.render_mode is not None:
            self._init_pygame()
        self._engine.get_occupancy_obs(out=self._frame_buf)
        frame_u8 = (self._frame_buf * 255.0).astype(np.uint8)
        for i in range(self.frame_skip):
            ch0 = i * GameEngine.OCCUPANCY_CHANNELS
            ch1 = ch0 + GameEngine.OCCUPANCY_CHANNELS
            self._obs_buf[:, :, ch0:ch1] = np.transpose(frame_u8, (1, 2, 0))
        return self._obs_buf.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False
        truncated = False

        for i in range(self.frame_skip):
            reward, terminated, truncated = self._engine.step(int(action))
            total_reward += reward
            self._engine.get_occupancy_obs(out=self._frame_buf)
            ch0 = i * GameEngine.OCCUPANCY_CHANNELS
            ch1 = ch0 + GameEngine.OCCUPANCY_CHANNELS
            self._obs_buf[:, :, ch0:ch1] = np.transpose(
                (self._frame_buf * 255.0).astype(np.uint8), (1, 2, 0)
            )
            if self.render_mode == "human":
                self.render()
            if terminated or truncated:
                break

        info = {
            "score": self._engine.score,
            "lives": self._engine.player.lives,
            "segments_destroyed": self._engine.segments_destroyed,
            "spiders_destroyed": self._engine.spiders_destroyed,
        }
        return self._obs_buf.copy(), float(total_reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        self._init_pygame()
        self._engine.render(self._surf, self._font)

        if self.render_mode == "human":
            pygame.event.pump()
            if self.window_size != (WIDTH, HEIGHT):
                self._window.fill((14, 16, 22))
            self._window.blit(self._surf, (0, 0))
            if self.present_fn is not None:
                self.present_fn(self._window)
            pygame.display.flip()
            self._clock.tick(self.render_fps)
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
