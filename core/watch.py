"""
Watch the trained Double DQN agent play Centipede.

Usage:
    uv run python -m core.watch [--model PATH] [--episodes N] [--fps N]

PATH is the model path without .zip (e.g. models/dqn_centipede or
models/dqn_centipede_ckpt_300000_steps). Run with no --model to pick
interactively from saved models.
"""
from __future__ import annotations

import argparse
import os

import pygame
from core.env import CentipedeEnv
from core.game import WIDTH, HEIGHT
from core.train import MODEL_PATH, DoubleDQN, _make_env_fn, list_saved_models


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


def watch(model_path: str = MODEL_PATH, episodes: int = 5, fps: int = 60) -> None:
    if not os.path.exists(model_path + ".zip"):
        print(f"No model found at {model_path}.zip — train first.")
        return

    model = DoubleDQN.load(model_path)
    print(f"Loaded model from {model_path}.zip")

    proc_env = _make_env_fn(seed=0)()

    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Centipede – DDQN Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    surf = pygame.Surface((WIDTH, HEIGHT))

    base_env: CentipedeEnv = proc_env.unwrapped

    for ep in range(episodes):
        obs, _ = proc_env.reset()
        total_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    proc_env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    proc_env.close()
                    pygame.quit()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = proc_env.step(int(action))
            total_reward += reward
            done = terminated or truncated

            base_env._engine.render(surf, font)
            window.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        print(f"Episode {ep + 1}: score={info['score']}  total_reward={total_reward:.0f}")

    proc_env.close()
    pygame.quit()


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
