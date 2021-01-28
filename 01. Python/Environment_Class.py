# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Reinforcement Learning Environment for 2048 Puzzle Game
------------------------------------------------------------------------------------------------------------------------
Date: January 28 of 2021
Considerations:
    -
Authors: Ignacio Brottier González
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importing libraries and necessary functions
# ----------------------------------------------------------------------------------------------------------------------
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

# ----------------------------------------------------------------------------------------------------------------------
# Open AI RL Environment Class
# ----------------------------------------------------------------------------------------------------------------------


class TwentyFortyEightEnvironment(Env):
    def __init__(self):
        # Action Space : Down, Left, Up, Right
        self.action_space = Discrete(4)

        # Observation Space : 4x4 Matrix
        self.observation_space = Box(
            low=np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            high=np.array([[999999, 999999, 999999, 999999],
                           [999999, 999999, 999999, 999999],
                           [999999, 999999, 999999, 999999],
                           [999999, 999999, 999999, 999999]]),
            shape=(4, 4),
            dtype=int)

        # Starting State : Zero survived turns -> objective is to survive as many turns as possible
        self.state = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        self.turn = 0

    def step(self, action):
        self.turn += 1
        reward = self.turn  # Reward = number of turns survived

        self.generate_new_number()
        done = self.check_if_done()

    def generate_new_number(self):
        """
        Finds all empty cells in the matrix and fills on of them with either 2 or 4
        """
        new_number = random.randrange(1, 3, 1) * 2

        empty_positions = []
        for row in range(len(self.observation_space.shape[0])):
            for col in range(len(self.observation_space[1])):
                if self.observation_space[row][col] == 0:
                    empty_positions.append([row, col])

        position = empty_positions[random.randrange(0, len(empty_positions) + 1, 1)]

        self.observation_space[position[0]][position[1]] = new_number

    def check_if_done(self):
        """
        Check if there's any available move to continue playing
        :return: done: bool
        """
        done = True
        for row in range(len(self.observation_space.shape[0])):
            for col in range(len(self.observation_space[1])):

                # If theres an empty cell, the game is not over
                if self.observation_space[row][col] == 0:
                    done = False

                if row > 0:
                    if self.observation_space[row][col] == self.observation_space[row - 1][col]:
                        done = False
                if row < self.observation_space.shape[0]:
                    if self.observation_space[row][col] == self.observation_space[row + 1][col]:
                        done = False
                if col > 0:
                    if self.observation_space[row][col] == self.observation_space[row][col - 1]:
                        done = False
                if col < self.observation_space.shape[1]:
                    if self.observation_space[row][col] == self.observation_space[row][col + 1]:
                        done = False

        return done

    def render(self, mode='human'):
        pass

    def reset(self):
        pass

