# Maze Navigation. Originally proposed in
# Backpropamine: differentiable neuromdulated plasticity.
#
# This code implements the "Grid Maze" task. See Section 4.5 in Miconi et al.
# ICML 2018 ( https://arxiv.org/abs/1804.02464 ), or Section 4.2 in
# Miconi et al. ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym )
#
# This file is modified to implement the maze itself. The `run.py` file
# contains the WarpGrad Network used in
#
# Meta-Learning With Warped Gradient Descent
# Flennerhag et. al., ICLR (2020), https://openreview.net/forum?id=rkeiQlBFPB

import collections

import numpy as np


Position = collections.namedtuple("Positions",
                                  ["right", "center",
                                   "reward_right", "reward_center"])

ObsSpec = collections.namedtuple("ObsSpec",
                                 ["num_action",
                                  "ref_size",
                                  "additional_inputs",
                                  "total_inputs",
                                  "episode_length"])


def is_reward(position, idx):
    """Is agent in rewarding position?

    Args:
        position (Position): current agent position
        idx (int): batch index.

    Returns:
        is_reward (bool): whether the agent is in a reward state or not.
    """
    return (position.reward_right[idx] == position.right[idx]) and \
           (position.reward_center[idx] == position.center[idx])


def initialize(maze_size, batch_size):
    """Initialize maze.

    Args:
        maze_size (int): size of maze (H, W).
        batch_size (int): number of parallel mazes.

    Returns (2):
        maze (np.array): grid maze.
        position (Position): the agent's position.
    """
    maze = np.ones((maze_size, maze_size))
    center = maze_size // 2

    # Grid maze
    maze[1:maze_size - 1, 1:maze_size - 1].fill(0)
    for row in range(1, maze_size - 1):
        for col in range(1, maze_size - 1):
            if row % 2 == 0 and col % 2 == 0:
                maze[row, col] = 1
    maze[center, center] = 0

    pos_right = {}
    pos_center = {}
    pos_reward_right = {}
    pos_reward_center = {}
    for nb in range(batch_size):
        # Note: it doesn't matter if the reward is on the center (see below).
        # All we need is not to put it on a wall or pillar (maze=1)
        myrposr = 0; myrposc = 0
        while (maze[myrposr, myrposc] == 1) or \
                (myrposr == center and myrposc == center):
            myrposr = np.random.randint(1, maze_size - 1)
            myrposc = np.random.randint(1, maze_size - 1)
        pos_reward_right[nb] = myrposr; pos_reward_center[nb] = myrposc

        # Agent always starts an episode from the center
        pos_center[nb] = center
        pos_right[nb] = center

    return maze, Position(pos_center, pos_right,
                         pos_reward_center, pos_reward_right)


def step_fun(maze, position, actions, batch_size, wall_penalty, reward_value):
    """Step function for maze

    Args:
        maze (np.array): the underlying maze.
        position (Position): current agent position.
        actions (np.array): actions taken.
        batch_size (int): num parallel envs.
        wall_penalty (float): penalty for hitting walls.
        reward_value (float): value at goal location.

    Returns (2):
        new_position (Position): the agent's update position.
        rewards (np.array): rewards for each parallel env.
    """
    maze_size = maze.shape[0]
    reward = np.zeros(batch_size)

    for nb in range(batch_size):
        action = actions[nb]
        to_position_center = position.center[nb]
        to_position_right = position.right[nb]
        if action == 0:  # Up
            to_position_right -= 1
        elif action == 1:  # Down
            to_position_right += 1
        elif action == 2:  # Left
            to_position_center -= 1
        elif action == 3:  # Right
            to_position_center += 1
        else:
            raise ValueError("Wrong Action")

        reward[nb] = 0.0
        if maze[to_position_right][to_position_center] == 1:
            reward[nb] -= wall_penalty
        else:
            position.center[nb] = to_position_center
            position.right[nb] = to_position_right

        if is_reward(position, nb):
            reward[nb] += reward_value
            while is_reward(position, nb):
                position.right[nb] = np.random.randint(1, maze_size - 1)
                position.center[nb] = np.random.randint(1, maze_size - 1)
    return position, reward


class Maze:

    """Navigation Maze.

    Args:
        obs_spec (ObsSpec): observation specs.
        maze_size (int): size of maze (H, W).
        batch_size (int): number of parallel mazes.
        batch_size (int): num parallel envs.
        wall_penalty (float): penalty for hitting walls.
        reward_value (float): value at goal location.
    """

    def __init__(self, obs_spec, maze_size, batch_size,
                 wall_penalty, reward_value):
        self._obs_spec = obs_spec
        self._batch_size = batch_size
        self._maze, self._position = initialize(maze_size, batch_size)

        def _step(actions):
            return step_fun(self._maze, self._position, actions,
                            batch_size, wall_penalty, reward_value)

        self._step = _step

    def step(self, actions):
        self._position, rewards = self._step(actions)
        return rewards

    def obs(self, actions, rewards, num_steps):
        tot_size = self._obs_spec.total_inputs
        ext_size = self._obs_spec.additional_inputs
        eps_size = self._obs_spec.episode_length
        ref_size = self._obs_spec.ref_size
        pos_size = ref_size * ref_size

        pos = self._position
        obs = np.zeros((self._batch_size, tot_size), dtype=np.float32)

        def get_right(idx):
            return (pos.right[idx] - ref_size // 2,
                    pos.right[idx] + ref_size // 2 + 1)

        def get_center(idx):
            return (pos.center[idx] - ref_size // 2,
                    pos.center[idx] + ref_size // 2 + 1)

        mz = self._maze.copy()
        for nb in range(self._batch_size):
            # Position
            x0, x1 = get_right(nb)
            y0, y1 = get_center(nb)
            obs[nb, 0:pos_size] = mz[x0:x1,y0:y1].flatten() * 1.0

            # Auxiliary inputs
            obs[nb, pos_size + 1] = 1.0  # Bias neuron
            obs[nb, pos_size + 2] = num_steps / eps_size
            obs[nb, pos_size + 3] = 1.0 * rewards[nb]
            obs[nb, pos_size + ext_size + actions[nb]] = 1
        return obs
