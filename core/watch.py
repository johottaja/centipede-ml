"""
Watch the trained C51 agent play Centipede.

Usage:
    uv run python -m core.watch [--model PATH] [--episodes N] [--fps N]

PATH is the model path without .zip (e.g. models/dqn_centipede or
models/dqn_centipede_ckpt_300000_steps). Run with no --model to pick
interactively from saved models.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pygame
import torch as th
import torch.nn.functional as F
from core.env import CentipedeEnv
from core.c51 import C51
from core.game import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    HEIGHT,
    NUM_ACTIONS,
    TILE,
    WIDTH,
    GameEngine,
)
from core.train import MODEL_PATH, FRAME_SKIP, list_saved_models

SIDEBAR_W = 168
ACTION_MAP_H = 148
WIN_W = WIDTH + SIDEBAR_W
WIN_H = HEIGHT + ACTION_MAP_H

PANEL_BG = (14, 16, 22)
PANEL_LINE = (42, 48, 62)
TEXT = (220, 226, 236)
TEXT_DIM = (130, 138, 152)
SELECTED_BORDER = (255, 220, 90)
BAR_BG = (28, 32, 42)

CHANNEL_META = [
    ("player", (0, 200, 255)),
    ("mushrooms", (50, 200, 80)),
    ("heads", (255, 70, 70)),
    ("body", (220, 40, 40)),
    ("spiders", (255, 165, 40)),
    ("bullet", (255, 255, 110)),
]

ACTION_SHORT = {
    0: "·",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓",
    5: "←F",
    6: "→F",
    7: "↑F",
    8: "↓F",
}
ACTION_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
    5: "LEFT+F",
    6: "RIGHT+F",
    7: "UP+F",
    8: "DOWN+F",
}


def _pick_model_interactive() -> str | None:
    models = list_saved_models()
    if not models:
        return None
    if len(models) == 1:
        return models[0][0]
    print("Available models:")
    for i, (path, label) in enumerate(models, 1):
        print(f"  {i}. {label}  ({path}.zip)")
    while True:
        choice = input(f"Choose [1-{len(models)}] (default 1): ").strip()
        if not choice:
            return models[0][0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1][0]
        print("Invalid choice.")


def _predict_action(model: C51, obs: np.ndarray) -> tuple[int, np.ndarray]:
    """Greedy C51 action plus a Boltzmann distribution over Q-values."""
    model.policy.set_training_mode(False)
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        q = model.q_net(obs_t).flatten()
        q_np = q.detach().cpu().numpy()
        span = float(q_np.max() - q_np.min())
        temperature = max(span / 4.0, 1.0)
        probs = F.softmax(q / temperature, dim=0).detach().cpu().numpy()
        action = int(q.argmax().item())
    return action, probs


class WatchOverlay:
    """Channel overlay + action map drawn around the unchanged playfield."""

    def __init__(self, env: CentipedeEnv, n_stack: int):
        self.env = env
        self.n_stack = n_stack
        self.n_ch = GameEngine.OCCUPANCY_CHANNELS
        self.quit = False
        self.channels_on = [True] * self.n_ch
        self.frames_on = [i == n_stack - 1 for i in range(n_stack)]
        self.action = ACTION_NOOP
        self.probs = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
        self._channel_rects: list[pygame.Rect] = []
        self._frame_rects: list[pygame.Rect] = []
        self._font: pygame.font.Font | None = None
        self._font_sm: pygame.font.Font | None = None
        self._overlay_surf: pygame.Surface | None = None

    def set_action(self, action: int, probs: np.ndarray) -> None:
        self.action = int(action)
        self.probs = np.asarray(probs, dtype=np.float32)

    def _fonts(self) -> tuple[pygame.font.Font, pygame.font.Font]:
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 14, bold=True)
            self._font_sm = pygame.font.SysFont("monospace", 12)
        return self._font, self._font_sm

    def present(self, window: pygame.Surface) -> None:
        self._handle_events()
        self._draw_channel_overlay(window)
        self._draw_sidebar(window)
        self._draw_action_map(window)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit = True
                elif pygame.K_1 <= event.key <= pygame.K_6:
                    idx = event.key - pygame.K_1
                    self.channels_on[idx] = not self.channels_on[idx]
                elif pygame.K_F1 <= event.key <= pygame.K_F1 + self.n_stack - 1:
                    idx = event.key - pygame.K_F1
                    self.frames_on[idx] = not self.frames_on[idx]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for i, rect in enumerate(self._channel_rects):
                    if rect.collidepoint(pos):
                        self.channels_on[i] = not self.channels_on[i]
                for i, rect in enumerate(self._frame_rects):
                    if rect.collidepoint(pos):
                        self.frames_on[i] = not self.frames_on[i]

    def _draw_channel_overlay(self, window: pygame.Surface) -> None:
        if not any(self.channels_on) or not any(self.frames_on):
            return
        if self._overlay_surf is None:
            self._overlay_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        obs = self.env._obs_buf
        surf = self._overlay_surf
        surf.fill((0, 0, 0, 0))
        px = pygame.surfarray.pixels3d(surf)
        pa = pygame.surfarray.pixels_alpha(surf)
        px[:] = 0
        pa[:] = 0

        any_on = False
        rgb = np.zeros((WIDTH, HEIGHT, 3), dtype=np.float32)
        alpha_f = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
        for fi, show_frame in enumerate(self.frames_on):
            if not show_frame:
                continue
            age = self.n_stack - 1 - fi
            frame_scale = 1.0 if age == 0 else max(0.35, 1.0 - 0.22 * age)
            base = fi * self.n_ch
            for ci, enabled in enumerate(self.channels_on):
                if not enabled:
                    continue
                grid = obs[:, :, base + ci]
                if not grid.any():
                    continue
                any_on = True
                color = CHANNEL_META[ci][1]
                scaled = np.repeat(np.repeat(grid, TILE, axis=0), TILE, axis=1).T
                strength = (scaled.astype(np.float32) / 255.0) * frame_scale
                rgb += strength[:, :, None] * np.array(color, dtype=np.float32)
                alpha_f = np.maximum(alpha_f, strength * 170.0)
        if any_on:
            np.clip(rgb, 0, 255, out=rgb)
            px[:] = rgb.astype(np.uint8)
            pa[:] = np.clip(alpha_f, 0, 170).astype(np.uint8)

        del px, pa
        if any_on:
            window.blit(surf, (0, 0))

    def _draw_sidebar(self, window: pygame.Surface) -> None:
        font, font_sm = self._fonts()
        x0 = WIDTH
        pygame.draw.rect(window, PANEL_BG, (x0, 0, SIDEBAR_W, HEIGHT))
        pygame.draw.line(window, PANEL_LINE, (x0, 0), (x0, HEIGHT), 1)

        y = 12
        title = font.render("CHANNELS", True, TEXT)
        window.blit(title, (x0 + 12, y))
        y += 22
        hint = font_sm.render("click / keys 1-6", True, TEXT_DIM)
        window.blit(hint, (x0 + 12, y))
        y += 22

        self._channel_rects = []
        for i, ((name, color), on) in enumerate(zip(CHANNEL_META, self.channels_on)):
            rect = pygame.Rect(x0 + 10, y, SIDEBAR_W - 20, 26)
            self._channel_rects.append(rect)
            pygame.draw.rect(window, (24, 28, 38) if on else (18, 20, 28), rect, border_radius=4)
            pygame.draw.rect(
                window,
                color if on else PANEL_LINE,
                rect,
                width=2 if on else 1,
                border_radius=4,
            )
            box = pygame.Rect(rect.x + 6, rect.y + 6, 14, 14)
            pygame.draw.rect(window, color if on else (40, 44, 54), box, border_radius=2)
            if on:
                pygame.draw.line(window, (10, 12, 16), (box.x + 3, box.y + 7), (box.x + 6, box.y + 11), 2)
                pygame.draw.line(window, (10, 12, 16), (box.x + 6, box.y + 11), (box.x + 11, box.y + 3), 2)
            label = font_sm.render(f"{i + 1}  {name}", True, TEXT if on else TEXT_DIM)
            window.blit(label, (rect.x + 26, rect.y + 6))
            y += 30

        y += 10
        title = font.render("STACK", True, TEXT)
        window.blit(title, (x0 + 12, y))
        y += 20
        hint = font_sm.render("F1-F4  newest last", True, TEXT_DIM)
        window.blit(hint, (x0 + 12, y))
        y += 20

        self._frame_rects = []
        for i, on in enumerate(self.frames_on):
            age = self.n_stack - 1 - i
            label = "t" if age == 0 else f"t-{age}"
            rect = pygame.Rect(x0 + 10, y, SIDEBAR_W - 20, 24)
            self._frame_rects.append(rect)
            pygame.draw.rect(window, (24, 28, 38) if on else (18, 20, 28), rect, border_radius=4)
            pygame.draw.rect(
                window,
                (90, 170, 255) if on else PANEL_LINE,
                rect,
                width=2 if on else 1,
                border_radius=4,
            )
            text = font_sm.render(f"F{i + 1}  {label}", True, TEXT if on else TEXT_DIM)
            window.blit(text, (rect.x + 10, rect.y + 5))
            y += 28

        y += 8
        note = font_sm.render("game shows", True, TEXT_DIM)
        window.blit(note, (x0 + 12, y))
        y += 14
        note = font_sm.render("through overlay", True, TEXT_DIM)
        window.blit(note, (x0 + 12, y))

    def _draw_action_map(self, window: pygame.Surface) -> None:
        font, font_sm = self._fonts()
        y0 = HEIGHT
        pygame.draw.rect(window, PANEL_BG, (0, y0, WIN_W, ACTION_MAP_H))
        pygame.draw.line(window, PANEL_LINE, (0, y0), (WIN_W, y0), 1)

        title = font.render("ACTION MAP", True, TEXT)
        window.blit(title, (12, y0 + 8))
        sel = font_sm.render(
            f"selected  {ACTION_NAMES[self.action]}",
            True,
            SELECTED_BORDER,
        )
        window.blit(sel, (140, y0 + 10))

        cell = 32
        gap = 4
        origin_x = 16
        origin_y = y0 + 36
        move_layout = {
            ACTION_UP: (1, 0),
            ACTION_LEFT: (0, 1),
            ACTION_NOOP: (1, 1),
            ACTION_RIGHT: (2, 1),
            ACTION_DOWN: (1, 2),
        }
        fire_layout = {
            7: (1, 0),
            5: (0, 1),
            6: (2, 1),
            8: (1, 2),
        }

        def draw_cell(action: int, col: int, row: int, ox: int) -> None:
            p = float(self.probs[action]) if action < len(self.probs) else 0.0
            x = ox + col * (cell + gap)
            y = origin_y + int(row * (cell + gap))
            rect = pygame.Rect(x, y, cell, cell)
            chosen = action == self.action
            pygame.draw.rect(window, BAR_BG, rect, border_radius=6)
            fill_h = int(round((cell - 6) * p))
            if fill_h > 0:
                fill = pygame.Rect(x + 3, y + cell - 3 - fill_h, cell - 6, fill_h)
                base = (70, 140, 255) if action < 5 else (255, 130, 70)
                if chosen:
                    base = tuple(min(255, c + 40) for c in base)
                pygame.draw.rect(window, base, fill, border_radius=4)
            pygame.draw.rect(
                window,
                SELECTED_BORDER if chosen else PANEL_LINE,
                rect,
                width=3 if chosen else 1,
                border_radius=6,
            )
            label = font_sm.render(ACTION_SHORT[action], True, TEXT)
            window.blit(
                label,
                (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2),
            )

        move_title = font_sm.render("move", True, TEXT_DIM)
        window.blit(move_title, (origin_x, y0 + 22))
        for action, (col, row) in move_layout.items():
            draw_cell(action, col, row, origin_x)

        fire_x = origin_x + 3 * (cell + gap) + 28
        fire_title = font_sm.render("move + fire", True, TEXT_DIM)
        window.blit(fire_title, (fire_x, y0 + 22))
        for action, (col, row) in fire_layout.items():
            draw_cell(action, col, row, fire_x)

        bars_x = fire_x + 3 * (cell + gap) + 24
        bars_w = WIN_W - bars_x - 16
        bar_h = 8
        bar_gap = 4
        start_y = y0 + 28
        for a in range(NUM_ACTIONS):
            p = float(self.probs[a])
            y = start_y + a * (bar_h + bar_gap)
            label = font_sm.render(ACTION_NAMES[a], True, SELECTED_BORDER if a == self.action else TEXT_DIM)
            window.blit(label, (bars_x, y - 2))
            track = pygame.Rect(bars_x + 62, y, bars_w - 62, bar_h)
            pygame.draw.rect(window, BAR_BG, track, border_radius=3)
            fill_w = max(1, int(track.w * p)) if p > 0 else 0
            if fill_w:
                color = (255, 210, 80) if a == self.action else (
                    (70, 140, 255) if a < 5 else (255, 130, 70)
                )
                pygame.draw.rect(
                    window,
                    color,
                    pygame.Rect(track.x, track.y, fill_w, track.h),
                    border_radius=3,
                )
            if a == self.action:
                pygame.draw.rect(window, SELECTED_BORDER, track, width=1, border_radius=3)


def watch(model_path: str = MODEL_PATH, episodes: int = 5, fps: int = 60) -> None:
    if not os.path.exists(model_path + ".zip"):
        print(f"No model found at {model_path}.zip — train first.")
        return

    model = C51.load(model_path)
    print(f"Loaded model from {model_path}.zip")

    env = CentipedeEnv(
        render_mode="human",
        frame_skip=FRAME_SKIP,
        window_size=(WIN_W, WIN_H),
    )
    overlay = WatchOverlay(env, n_stack=FRAME_SKIP)
    env.present_fn = overlay.present
    env.render_fps = fps
    obs, _ = env.reset()
    pygame.display.set_caption("Centipede – C51 Agent")

    for ep in range(episodes):
        if overlay.quit:
            break
        if ep > 0:
            obs, _ = env.reset()
        total_reward = 0.0
        done = False
        info: dict = {}

        while not done:
            if overlay.quit:
                env.close()
                return

            action, probs = _predict_action(model, obs)
            overlay.set_action(action, probs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: score={info['score']}  total_reward={total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model path without .zip (default: final model, or prompt if missing)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    model_path = args.model
    if model_path is None:
        if os.path.exists(MODEL_PATH + ".zip"):
            model_path = MODEL_PATH
        else:
            model_path = _pick_model_interactive()
            if model_path is None:
                print("No models found — train first.")
                raise SystemExit(1)
    watch(model_path=model_path, episodes=args.episodes, fps=args.fps)
