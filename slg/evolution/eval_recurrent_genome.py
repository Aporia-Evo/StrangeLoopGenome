import neat
import numpy as np

from slg.envs.gridworld import GridWorld
from slg.energy.energy import total_energy


def evaluate_recurrent_genome(genome, config, episodes=5, return_details=False):
    total_reward = 0.0
    total_energy_score = 0.0
    total_foods = 0
    total_wall_hits = 0

    for episode in range(episodes):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        env = GridWorld(size=8, max_steps=96, seed=episode)

        obs = env.reset()
        done = False

        actions = []
        activations = []

        while not done:
            output = net.activate(obs)
            activations.extend(output)

            action = int(np.argmax(output))
            actions.append(action)

            obs, reward, done, info = env.step(action)
            total_reward += reward

        action_entropy = len(set(actions)) / 5.0
        temporal_variance = np.var(activations) if activations else 0.0

        energy = total_energy(
            distance_to_food=info['distance'],
            agent_energy=info['energy'],
            action_entropy=action_entropy,
            num_nodes=len(genome.nodes),
            num_connections=len(genome.connections),
            active_connections=sum(
                1 for c in genome.connections.values() if c.enabled
            ),
            activations=activations,
        )

        energy += 0.01 * temporal_variance
        energy += 0.005 * info['wall_hits']

        total_energy_score += energy
        total_foods += info['foods_collected']
        total_wall_hits += info['wall_hits']

    avg_reward = total_reward / episodes
    avg_energy = total_energy_score / episodes
    avg_foods = total_foods / episodes
    avg_wall_hits = total_wall_hits / episodes

    task_score = avg_reward + 2.0 * avg_foods
    fitness = task_score - 0.35 * avg_energy

    if return_details:
        return {
            'fitness': float(fitness),
            'avg_reward': float(avg_reward),
            'avg_energy': float(avg_energy),
            'avg_foods': float(avg_foods),
            'avg_wall_hits': float(avg_wall_hits),
            'task_score': float(task_score),
        }

    return float(fitness)
