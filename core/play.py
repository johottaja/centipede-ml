"""
Human-playable runner.  Boots CentipedeEnv in "human" render mode and
translates keyboard input into discrete actions each frame.
"""
import sys
import pygame
from core.env import CentipedeEnv
from core.game import (
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN,
    NUM_MOVES,
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
    return move + (NUM_MOVES if fire else 0)


def main():
    env = CentipedeEnv(render_mode="human", frame_skip=1)
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
