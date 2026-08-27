"""
Human-playable runner.  Boots CentipedeEnv in "human" render mode and
translates keyboard input into discrete actions each frame.
"""
import sys
import pygame
from core.env import CentipedeEnv
from core.hparams import env_reward_kwargs
from core.game import (
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN,
    ACTION_FIRE, NUM_MOVES,
)


def keys_to_action(keys) -> int:
    left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    up    = keys[pygame.K_UP]    or keys[pygame.K_w]
    down  = keys[pygame.K_DOWN]  or keys[pygame.K_s]
    fire  = keys[pygame.K_SPACE]

    if left and not right:
        move = ACTION_LEFT
    elif right and not left:
        move = ACTION_RIGHT
    elif up and not down:
        move = ACTION_UP
    elif down and not up:
        move = ACTION_DOWN
    else:
        move = ACTION_NOOP
    if fire and move != ACTION_NOOP:
        return NUM_MOVES + (move - 1)
    if fire:
        return ACTION_FIRE
    return move


def main():
    env = CentipedeEnv(render_mode="human", frame_skip=1, **env_reward_kwargs())
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
        else:
            env.render()


if __name__ == "__main__":
    main()
