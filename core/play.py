"""
Human-playable runner.  Boots CentipedeEnv in "human" render mode and
translates keyboard input into discrete actions each frame.
"""
import sys
import pygame
from core.env import CentipedeEnv
from core.game import (
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN,
    ACTION_FIRE, ACTION_LEFT_FIRE, ACTION_RIGHT_FIRE,
    ACTION_UP_FIRE, ACTION_DOWN_FIRE,
)


def keys_to_action(keys) -> int:
    left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    up    = keys[pygame.K_UP]    or keys[pygame.K_w]
    down  = keys[pygame.K_DOWN]  or keys[pygame.K_s]
    fire  = keys[pygame.K_SPACE]

    if left  and fire: return ACTION_LEFT_FIRE
    if right and fire: return ACTION_RIGHT_FIRE
    if up    and fire: return ACTION_UP_FIRE
    if down  and fire: return ACTION_DOWN_FIRE
    if fire:           return ACTION_FIRE
    if left:           return ACTION_LEFT
    if right:          return ACTION_RIGHT
    if up:             return ACTION_UP
    if down:           return ACTION_DOWN
    return ACTION_NOOP


def main():
    env = CentipedeEnv(render_mode="human")
    env.reset()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                env.close()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r and env._engine.terminated:
                    env.reset()

        action = keys_to_action(pygame.key.get_pressed())

        if not env._engine.terminated:
            env.step(action)

        env.render()


if __name__ == "__main__":
    main()
