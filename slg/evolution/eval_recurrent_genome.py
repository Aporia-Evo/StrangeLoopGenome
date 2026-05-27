import neat
import numpy as np

from slg.envs.gridworld import GridWorld
from slg.energy.energy import total_energy


def settle(net, obs, inner_steps):
    """Run the recurrent net for inner_steps iterations on the same obs.

    Returns (final_output, mean_delta) where mean_delta is the average
    squared L2 distance between consecutive outputs during settling -
    smaller delta means the network converged to a stable attractor.
    """
    deltas = []
    prev = None
    output = None
    for _ in range(inner_steps):
        output = net.activate(obs)
        if prev is not None:
            delta = sum((a - b) ** 2 for a, b in zip(output, prev))
            deltas.append(delta)
        prev = list(output)

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return output, mean_delta


def evaluate_recurrent_genome(
    genome,
    config,
    episodes=5,
    return_details=False,
    energy_weight=0.35,
    inner_steps=1,
    convergence_weight=0.0,
):
    total_reward = 0.0
    total_energy_score = 0.0
    total_foods = 0
    total_wall_hits = 0
    total_no_progress = 0
    total_progress = 0.0
    total_settle_delta = 0.0
    settle_step_count = 0

    for episode in range(episodes):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        env = GridWorld(size=8, max_steps=96, seed=episode)

        obs = env.reset()
        done = False

        actions = []
        activations = []

        while not done:
            if inner_steps > 1:
                output, mean_delta = settle(net, obs, inner_steps)
                total_settle_delta += mean_delta
                settle_step_count += 1
            else:
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
        energy += 0.01 * info['wall_hits']

        total_energy_score += energy
        total_foods += info['foods_collected']
        total_wall_hits += info['wall_hits']
        total_no_progress += info['no_progress_steps']
        total_progress += info['total_progress']

    avg_reward = total_reward / episodes
    avg_energy = total_energy_score / episodes
    avg_foods = total_foods / episodes
    avg_wall_hits = total_wall_hits / episodes
    avg_no_progress = total_no_progress / episodes
    avg_progress = total_progress / episodes
    mean_settle_delta = (
        total_settle_delta / settle_step_count if settle_step_count else 0.0
    )

    task_score = avg_reward + 2.0 * avg_foods
    fitness = (
        task_score
        - energy_weight * avg_energy
        - 0.03 * avg_wall_hits
        - 0.01 * avg_no_progress
        - convergence_weight * mean_settle_delta
    )

    if return_details:
        return {
            'fitness': float(fitness),
            'avg_reward': float(avg_reward),
            'avg_energy': float(avg_energy),
            'avg_foods': float(avg_foods),
            'avg_wall_hits': float(avg_wall_hits),
            'avg_no_progress': float(avg_no_progress),
            'avg_progress': float(avg_progress),
            'task_score': float(task_score),
            'mean_settle_delta': float(mean_settle_delta),
            'inner_steps': int(inner_steps),
        }

    return float(fitness)
