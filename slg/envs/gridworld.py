import numpy as np

class GridWorld:
    ACTIONS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
        4: (0, 0),
    }

    def __init__(self, size=8, max_steps=64, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_energy = 1.0
        self.agent_pos = np.array([
            self.rng.integers(0, self.size),
            self.rng.integers(0, self.size),
        ])
        self.food_pos = np.array([
            self.rng.integers(0, self.size),
            self.rng.integers(0, self.size),
        ])
        return self.observe()

    def observe(self):
        delta = (self.food_pos - self.agent_pos) / max(1, self.size - 1)
        return np.array([
            delta[0],
            delta[1],
            self.agent_energy,
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        dx, dy = self.ACTIONS[int(action)]
        self.agent_pos = np.clip(
            self.agent_pos + np.array([dx, dy]),
            0,
            self.size - 1,
        )

        self.agent_energy -= 0.01

        distance = np.linalg.norm(self.food_pos - self.agent_pos, ord=1)
        reward = -0.01 * distance

        reached_food = np.array_equal(self.agent_pos, self.food_pos)

        if reached_food:
            reward += 1.0

        done = self.steps >= self.max_steps or self.agent_energy <= 0.0

        return self.observe(), reward, done, {
            'distance': distance,
            'energy': self.agent_energy,
            'reached_food': reached_food,
        }
