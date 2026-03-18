"""
Watch the trained DQN agent play Centipede.

Usage:
    uv run python watch.py [--model PATH] [--episodes N] [--fps N]
"""
from __future__ import annotations

import argparse
import os

import pygame
from stable_baselines3 import DQN

from env import CentipedeEnv
from train import MODEL_PATH, make_env


def watch(model_path: str = MODEL_PATH, episodes: int = 5, fps: int = 60) -> None:
    if not os.path.exists(model_path + ".zip"):
        print(f"No model found at {model_path}.zip — train first.")
        return

    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}.zip")

    proc_env = make_env()

    # Reach the underlying CentipedeEnv to grab its engine for rendering
    from game import WIDTH, HEIGHT
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Centipede – DQN Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    surf = pygame.Surface((WIDTH, HEIGHT))

    # Unwrap to CentipedeEnv
    base_env = proc_env.unwrapped

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
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    watch(model_path=args.model, episodes=args.episodes, fps=args.fps)
