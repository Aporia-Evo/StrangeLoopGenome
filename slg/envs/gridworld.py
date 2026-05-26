import numpy as np


class GridWorld:
    ACTIONS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
        4: (0, 0),
    }

    def __init__(self, size=8, max_steps=96, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _sample_empty_pos(self, forbidden=None):
        forbidden = [] if forbidden is None else forbidden

        while True:
            pos = np.array([
                self.rng.integers(0, self.size),
                self.rng.integers(0, self.size),
            ])

            if not any(np.array_equal(pos, f) for f in forbidden):
                return pos

    def _distance_to_food(self):
        return np.linalg.norm(self.food_pos - self.agent_pos, ord=1)

    def reset(self):
        self.steps = 0
        self.agent_energy = 1.0
        self.wall_hits = 0
        self.foods_collected = 0
        self.no_progress_steps = 0
        self.total_progress = 0.0

        self.agent_pos = self._sample_empty_pos()
        self.food_pos = self._sample_empty_pos(forbidden=[self.agent_pos])
        self.prev_distance = self._distance_to_food()

        return self.observe()

    def observe(self):
        denom = max(1, self.size - 1)
        delta = (self.food_pos - self.agent_pos) / denom

        x_norm = self.agent_pos[0] / denom
        y_norm = self.agent_pos[1] / denom

        return np.array([
            delta[0],
            delta[1],
            self.agent_energy,
            x_norm,
            y_norm,
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1

        prev_distance = self._distance_to_food()

        dx, dy = self.ACTIONS[int(action)]
        proposed_pos = self.agent_pos + np.array([dx, dy])

        hit_wall = bool(np.any(proposed_pos < 0) or np.any(proposed_pos >= self.size))

        if hit_wall:
            self.wall_hits += 1
            new_pos = self.agent_pos
        else:
            new_pos = proposed_pos

        self.agent_pos = new_pos

        is_wait = int(action) == 4
        move_cost = 0.004 if is_wait else 0.01
        wall_cost = 0.03 if hit_wall else 0.0
        self.agent_energy -= move_cost + wall_cost

        distance = self._distance_to_food()
        progress = prev_distance - distance
        self.total_progress += progress

        if progress <= 0:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        reward = -0.005 * distance + 0.04 * progress

        reached_food = np.array_equal(self.agent_pos, self.food_pos)

        if reached_food:
            self.foods_collected += 1
            reward += 1.0
            self.agent_energy = min(1.0, self.agent_energy + 0.25)
            self.food_pos = self._sample_empty_pos(forbidden=[self.agent_pos])
            distance = self._distance_to_food()
            self.prev_distance = distance
            self.no_progress_steps = 0
        else:
            self.prev_distance = distance

        done = self.steps >= self.max_steps or self.agent_energy <= 0.0

        return self.observe(), reward, done, {
            'distance': distance,
            'energy': self.agent_energy,
            'hit_wall': hit_wall,
            'wall_hits': self.wall_hits,
            'reached_food': reached_food,
            'foods_collected': self.foods_collected,
            'progress': progress,
            'total_progress': self.total_progress,
            'no_progress_steps': self.no_progress_steps,
        }
