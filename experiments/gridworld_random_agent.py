import numpy as np

from slg.envs.gridworld import GridWorld
from slg.energy.energy import total_energy


def run_episode(env):
    obs = env.reset()

    total_reward = 0.0
    actions = []

    done = False

    while not done:
        action = np.random.randint(0, 5)
        actions.append(action)

        obs, reward, done, info = env.step(action)
        total_reward += reward

    action_entropy = len(set(actions)) / 5.0

    energy = total_energy(
        distance_to_food=info['distance'],
        agent_energy=info['energy'],
        action_entropy=action_entropy,
    )

    return {
        'reward': total_reward,
        'energy': energy,
        'steps': env.steps,
    }


if __name__ == '__main__':
    env = GridWorld(size=8, max_steps=64)

    results = []

    for episode in range(10):
        result = run_episode(env)
        results.append(result)

        print(
            f"Episode {episode} | "
            f"reward={result['reward']:.3f} | "
            f"energy={result['energy']:.3f} | "
            f"steps={result['steps']}"
        )
