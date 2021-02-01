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
        # 1. Apply Action
        if action == 0:
            self.action_down()
        elif action == 1:
            self.action_left()
        elif action == 2:
            self.action_up()
        elif action == 3:
            self.action_right()

        # 2. Generate new number
        self.generate_new_number()

        # 3. Check if done
        done = self.check_if_done()

        # 4. Update reward
        self.turn += 1
        reward = max(self.state)  # Reward = number of turns survived
        self.observation_space = self.state

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def action_up(self):
        for col in range(self.state.shape[1]):
            row = 3
            while row-1 >= 0:
                if self.state[row - 1][col] == 0:
                    self.state[row - 1][col] = self.state[row][col]
                    self.state[row][col] = 0
                elif self.state[row - 1][col] == self.state[row][col]:
                    self.state[row - 1][col] = self.state[row - 1][col] * 2
                    self.state[row][col] = 0
                row -= 1

    def action_left(self):
        for row in range(self.state.shape[0]):
            col = 3
            while col-1 >= 0:
                if self.state[row][col-1] == 0:
                    self.state[row][col-1] = self.state[row][col]
                    self.state[row][col] = 0
                elif self.state[row][col-1] == self.state[row][col]:
                    self.state[row][col-1] = self.state[row][col-1] * 2
                    self.state[row][col] = 0
                col -= 1

    def action_right(self):
        for row in range(self.state.shape[0]):
            col = 0
            while col+1 <= 3:
                if self.state[row][col + 1] == 0:
                    self.state[row][col + 1] = self.state[row][col]
                    self.state[row][col] = 0
                elif self.state[row][col + 1] == self.state[row][col]:
                    self.state[row][col + 1] = self.state[row][col + 1] * 2
                    self.state[row][col] = 0
                col += 1

    def action_down(self):
        for col in range(self.state.shape[1]):
            row = 0
            while row+1 <= 3:
                if self.state[row + 1][col] == 0:
                    self.state[row + 1][col] = self.state[row][col]
                    self.state[row][col] = 0
                elif self.state[row + 1][col] == self.state[row][col]:
                    self.state[row + 1][col] = self.state[row + 1][col] * 2
                    self.state[row][col] = 0
                row += 1

    def generate_new_number(self):
        """
        Finds all empty cells in the matrix and fills one of them (randomly) with either 2 or 4
        """
        new_number = random.randrange(1, 3, 1) * 2

        empty_positions = []
        for row in range(self.state.shape[0]):
            for col in range(self.state.shape[1]):
                if self.state[row][col] == 0:
                    empty_positions.append([row, col])

        if len(empty_positions) > 0:
            position = empty_positions[random.randrange(0, len(empty_positions), 1)]
            self.state[position[0]][position[1]] = new_number
        else:
            pass

    def check_if_done(self, done = True):
        """
        Check if there's any available move to continue playing
        :return: done: bool
        """
        for row in range(self.state.shape[0]):
            for col in range(self.state.shape[1]):

                # If theres an empty cell, the game is not over
                if self.state[row][col] == 0:
                    done = False

                if row > 0:
                    if self.state[row][col] == self.state[row - 1][col]:
                        done = False
                if row < self.state.shape[0]-1:
                    if self.state[row][col] == self.state[row + 1][col]:
                        done = False
                if col > 0:
                    if self.state[row][col] == self.state[row][col - 1]:
                        done = False
                if col < self.state.shape[1]-1:
                    if self.state[row][col] == self.state[row][col + 1]:
                        done = False

        return done

    def render(self, mode='human'):
        print(self.state)

    def reset(self):
        self.state = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        self.turn = 0
        self.generate_new_number()
        return self.state
