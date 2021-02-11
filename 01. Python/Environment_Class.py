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
import math

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
            dtype=int
        )

        # Starting State : Zero survived turns -> objective is to survive as many turns as possible
        self.state = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        self.turn = 0
        self.last_action = -1
        self.info = {
            "D": 0,
            "L": 0,
            "U": 0,
            "R": 0,
            #'Max': 0
        }

    def step(self, action):
        # 1. Apply Action
        valid_actions = self._get_available_movements()
        invalid_movement = False

        if action in valid_actions:
            if action == 0:
                self.action_down()
                self.info['D'] += 1
            elif action == 1:
                self.action_left()
                self.info['L'] += 1
            elif action == 2:
                self.action_up()
                self.info['U'] += 1
            elif action == 3:
                self.action_right()
                self.info['R'] += 1

            # 2. Generate new number
            self.generate_new_number()

        else:   # Tried to apply an invalid movement
            invalid_movement = True

        # 3. Check if done
        done = self.check_if_done()
        self.observation_space = self.state
        
        # 4, Update Reward
        reward = -999 if invalid_movement else self._get_reward()

        # Set placeholder for info
        self.turn += 1
        self.last_action = action

        # Return step information
        return self.state, reward, done, self.info

    def _get_available_movements(self):
        valid_movements = []
        if self.action_down(apply_action=False):
            valid_movements.append(0)
        if self.action_left(apply_action=False):
            valid_movements.append(1)
        if self.action_up(apply_action=False):
            valid_movements.append(2)
        if self.action_right(apply_action=False):
            valid_movements.append(3)
        return valid_movements

    def _get_reward(self, option=0):
        aux = self.state.copy()
        max_aux = 0
        for row in range(aux.shape[0]):
            for col in range(aux.shape[1]):
                if aux[row][col] > max_aux:
                    max_aux = aux[row][col]
                aux[row][col] = math.sqrt(aux[row][col])
        
        #self.info["Max"] = max_aux

        if option == 0:
            reward = sum(sum(aux)) + math.sqrt(max_aux)/2
        elif option == 1:
            reward = sum(sum(self.state)) + max_aux
        elif option == 2:
            reward = max_aux
        else:
            reward = 1

        return reward

    def _get_columns(self, reverse):
        col0 = np.array([0, 0, 0, 0])
        col1 = np.array([0, 0, 0, 0])
        col2 = np.array([0, 0, 0, 0])
        col3 = np.array([0, 0, 0, 0])

        for row in range(self.state.shape[0]):
            col0[row] = self.state[row][0]
            col1[row] = self.state[row][1]
            col2[row] = self.state[row][2]
            col3[row] = self.state[row][3]

        if reverse:
            col1 = col1[::-1]
            col2 = col2[::-1]
            col3 = col3[::-1]
            col0 = col0[::-1]

        return col0, col1, col2, col3
    
    def _get_rows(self, reverse):
        row0 = self.state[0]
        row1 = self.state[1]
        row2 = self.state[2]
        row3 = self.state[3]
        
        if reverse:
            row0 = row0[::-1]
            row1 = row1[::-1]
            row2 = row2[::-1]
            row3 = row3[::-1]

        return row0, row1, row2, row3

    def _reconstruct_board(self, new_columns=None, new_rows=None):
        if new_columns is not None:
            col_index = -1
            for col in new_columns:
                col_index += 1
                for row in range(0, len(col)):
                    self.state[row][col_index] = col[row]

        if new_rows is not None:
            row_index = -1
            for row in new_rows:
                row_index += 1
                for col in range(0, len(row)):
                    self.state[row_index][col] = row[col]

    def action_up(self, apply_action=True):
        cols = [self._get_columns(reverse=False)]

        cols_1 = [[x[x != 0] for x in col] for col in cols]    # Columns without 0s
        cols_0 = [[x[x == 0] for x in col] for col in cols]    # Columns of 0s

        new_cols = []
        index = -1
        for col in cols_1[0]:
            index += 1
            col_0 = cols_0[0][index]
            if len(col) >= 2:
                for row in range(0, len(col)):
                    if col[row] == col[row-1]:
                        col[row-1] = col[row-1]*2
                        col[row] = 0
                    if col[row-1] == 0:
                        col[row-1] = col[row]
                        col[row] = 0

            new_cols.append(np.concatenate((col, col_0), axis=None))

        # is_valid_movement = not (set(cols) == set(new_cols))
        is_valid_movement = False
        for outer_index in range(len(cols[0])):
            old = cols[0][outer_index]
            new = new_cols[outer_index]
            for inner_index in range(len(old)):
                if old[inner_index] != new[inner_index]:
                    is_valid_movement = True

        if apply_action and is_valid_movement:
            self._reconstruct_board(new_columns=new_cols)

        return is_valid_movement

    def action_left(self, apply_action=True):
        rows = [self._get_rows(reverse=False)]

        rows_1 = [[x[x != 0] for x in row] for row in rows]  # Rows without 0s
        rows_0 = [[x[x == 0] for x in row] for row in rows]  # Rows of 0s

        new_rows = []
        index = -1
        for row in rows_1[0]:
            index += 1
            row_0 = rows_0[0][index]
            if len(row) >= 2:
                for col in range(0, len(row)):
                    if row[col] == row[col - 1]:
                        row[col - 1] = row[col - 1] * 2
                        row[col] = 0
                    if row[col - 1] == 0:
                        row[col - 1] = row[col]
                        row[col] = 0

            new_rows.append(np.concatenate((row, row_0), axis=None))

        # is_valid_movement = not (set(rows) == set(new_rows))
        is_valid_movement = False
        for outer_index in range(len(rows[0])):
            old = rows[0][outer_index]
            new = new_rows[outer_index]
            for inner_index in range(len(old)):
                if old[inner_index] != new[inner_index]:
                    is_valid_movement = True

        if apply_action and is_valid_movement:
            self._reconstruct_board(new_rows=new_rows)

        return is_valid_movement

    def action_right(self, apply_action=True):
        rows = [self._get_rows(reverse=True)]

        rows_1 = [[x[x != 0] for x in row] for row in rows]  # Rows without 0s
        rows_0 = [[x[x == 0] for x in row] for row in rows]  # Rows of 0s

        new_rows = []
        index = -1
        for row in rows_1[0]:
            index += 1
            row_0 = rows_0[0][index]
            if len(row) >= 2:
                for col in range(0, len(row)):
                    if row[col] == row[col - 1]:
                        row[col - 1] = row[col - 1] * 2
                        row[col] = 0
                    if row[col - 1] == 0:
                        row[col - 1] = row[col]
                        row[col] = 0

            new_rows.append(np.concatenate((row, row_0), axis=None))

        # is_valid_movement = not (set(rows) == set(new_rows))
        is_valid_movement = False
        for outer_index in range(len(rows[0])):
            old = rows[0][outer_index]
            new = new_rows[outer_index]
            for inner_index in range(len(old)):
                if old[inner_index] != new[inner_index]:
                    is_valid_movement = True

        for i in range(len(new_rows)):
            new_rows[i] = new_rows[i][::-1]

        if apply_action and is_valid_movement:
            self._reconstruct_board(new_rows=new_rows)

        return is_valid_movement

    def action_down(self, apply_action=True):
        cols = [self._get_columns(reverse=True)]

        cols_1 = [[x[x != 0] for x in col] for col in cols]    # Columns without 0s
        cols_0 = [[x[x == 0] for x in col] for col in cols]    # Columns of 0s

        new_cols = []
        index = -1
        for col in cols_1[0]:
            index += 1
            col_0 = cols_0[0][index]
            if len(col) >= 2:
                for row in range(0, len(col)):
                    if col[row] == col[row-1]:
                        col[row-1] = col[row-1]*2
                        col[row] = 0
                    if col[row-1] == 0:
                        col[row-1] = col[row]
                        col[row] = 0

            new_cols.append(np.concatenate((col, col_0), axis=None))

        # is_valid_movement = not (set(cols) == set(new_cols))
        is_valid_movement = False
        for outer_index in range(len(cols[0])):
            old = cols[0][outer_index]
            new = new_cols[outer_index]
            for inner_index in range(len(old)):
                if old[inner_index] != new[inner_index]:
                    is_valid_movement = True

        for i in range(len(new_cols)):
            new_cols[i] = new_cols[i][::-1]

        if apply_action and is_valid_movement:
            self._reconstruct_board(new_columns=new_cols)

        return is_valid_movement

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

        generated_number = 1
        if len(empty_positions) > 0:
            position = empty_positions[random.randrange(0, len(empty_positions), 1)]
            self.state[position[0]][position[1]] = new_number
        else:
            generated_number = -1
            #self.info['X'] += 1

        return generated_number

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
        print(self.turn)
        if self.last_action == 0:
            print('⬇⬇⬇ D ⬇⬇⬇')
        elif self.last_action == 1:
            print('⬅⬅⬅ L ⬅⬅⬅')
        elif self.last_action == 2:
            print('⬆⬆⬆ U ⬆⬆⬆')
        else:
            print('➡➡➡ R ➡➡➡')

        print(self.state)
        print(self.info)
        print('===========\n')

    def reset(self):
        self.state = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        self.turn = 0
        self.last_action = -1
        self.info = {
            "D": 0,
            "L": 0,
            "U": 0,
            "R": 0
        }
        for i in range(1):
            self.generate_new_number()

        return self.state
