import numpy as np


def greedy_action(obs):
    """
    Simple oracle policy for GridWorld.

    Observation layout:
    [delta_x, delta_y, energy, agent_x, agent_y]

    Actions:
    0 up, 1 down, 2 left, 3 right, 4 wait
    """
    dx = float(obs[0])
    dy = float(obs[1])

    if abs(dx) >= abs(dy):
        if dx < 0:
            return 2
        if dx > 0:
            return 3

    if dy < 0:
        return 0
    if dy > 0:
        return 1

    return 4


def random_action(rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.integers(0, 5))
