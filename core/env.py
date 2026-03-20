"""
Gymnasium environment for Centipede.

Observation : entity-centric float32 vector of length RELATIVE_OBS_SIZE (105)
              Layout:
                [0..83]   12 centipede segment slots × 7 features
                          (rel_x, rel_y, vel_x, vel_y, is_alive, is_head, dist_to_obstacle)
                [84..86]  bullet (rel_x, rel_y, is_alive)
                [87..96]  2 spider slots × 5 features
                          (rel_x, rel_y, vel_x, vel_y, is_alive)
                [97..104] 8-way lidar distances from player — walls + mushrooms only
                          (N, NE, E, SE, S, SW, W, NW)
Action space: Discrete(6)  – see game.ACTION_* constants
Reward       : score delta per step
Terminated   : player loses all lives
"""
from __future__ import annotations

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from core.game import (
    GameEngine,
    WIDTH, HEIGHT, NUM_ACTIONS,
)


class CentipedeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        reward_mushroom_hit: int = 1,
        reward_mushroom_destroy: int = 5,
        reward_body_hit: int = 10,
        reward_head_hit: int = 100,
        reward_depth_discount: float = 0.0,
        reward_depth_discount_fn: str = "linear",
        reward_spider_hit: int = 300,
        reward_spider_penalty: int = 0,
        reward_centipede_penalty: int = 0,
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), \
            f"Unsupported render_mode: {render_mode}"
        self.render_mode = render_mode

        # Entity-centric float32 vector; see module docstring for layout.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(GameEngine.RELATIVE_OBS_SIZE,),
            dtype=np.float32,
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
        )
        self._obs_buf = np.zeros(GameEngine.RELATIVE_OBS_SIZE, dtype=np.float32)
        self._surf: pygame.Surface | None = None   # off-screen surface
        self._window: pygame.Surface | None = None  # on-screen window (human mode)
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
        if self.render_mode is not None:
            self._init_pygame()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, terminated, truncated = self._engine.step(int(action))
        obs = self._get_obs()
        info = {
            "score": self._engine.score,
            "lives": self._engine.player.lives,
            "segments_destroyed": self._engine.segments_destroyed,
            "spiders_destroyed": self._engine.spiders_destroyed,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
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
        self._engine.get_relative_obs(out=self._obs_buf)
        return self._obs_buf.copy()
