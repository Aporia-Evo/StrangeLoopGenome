from pathlib import Path

import neat
import numpy as np
import torch

from slg.distill.recurrent_student import load_recurrent_student
from slg.distill.student import load_student
from slg.energy.energy import total_energy
from slg.envs.gridworld import GridWorld


class StudentPrior:
    def __init__(self, student_path):
        self.student_path = Path(student_path)
        checkpoint = torch.load(self.student_path, map_location='cpu')
        config = checkpoint.get('config', {})
        self.is_recurrent = 'seq_len' in config

        if self.is_recurrent:
            self.model, self.config, self.checkpoint = load_recurrent_student(
                self.student_path
            )
        else:
            self.model, self.config, self.checkpoint = load_student(
                self.student_path
            )

    def reset(self):
        if self.is_recurrent and hasattr(self.model, 'reset'):
            self.model.reset()

    def act(self, obs):
        return self.model.act(obs)


def evaluate_recurrent_genome_with_student_prior(
    genome,
    config,
    student_prior,
    episodes=5,
    prior_weight=1.0,
    return_details=False,
):
    total_reward = 0.0
    total_energy_score = 0.0
    total_foods = 0
    total_wall_hits = 0
    total_no_progress = 0
    total_progress = 0.0
    total_agreement = 0
    total_steps = 0

    for episode in range(episodes):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        env = GridWorld(size=8, max_steps=96, seed=episode)
        student_prior.reset()

        obs = env.reset()
        done = False
        actions = []
        activations = []
        agreements = 0
        steps = 0

        while not done:
            output = net.activate(obs)
            activations.extend(output)

            action = int(np.argmax(output))
            prior_action = int(student_prior.act(obs))

            if action == prior_action:
                agreements += 1

            actions.append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        action_entropy = len(set(actions)) / 5.0
        temporal_variance = np.var(activations) if activations else 0.0

        energy = total_energy(
            distance_to_food=info['distance'],
            agent_energy=info['energy'],
            action_entropy=action_entropy,
            num_nodes=len(genome.nodes),
            num_connections=len(genome.connections),
            active_connections=sum(1 for c in genome.connections.values() if c.enabled),
            activations=activations,
        )
        energy += 0.01 * temporal_variance
        energy += 0.01 * info['wall_hits']

        total_energy_score += energy
        total_foods += info['foods_collected']
        total_wall_hits += info['wall_hits']
        total_no_progress += info['no_progress_steps']
        total_progress += info['total_progress']
        total_agreement += agreements
        total_steps += steps

    avg_reward = total_reward / episodes
    avg_energy = total_energy_score / episodes
    avg_foods = total_foods / episodes
    avg_wall_hits = total_wall_hits / episodes
    avg_no_progress = total_no_progress / episodes
    avg_progress = total_progress / episodes
    prior_agreement = total_agreement / max(1, total_steps)

    task_score = avg_reward + 2.0 * avg_foods
    base_fitness = (
        task_score
        - 0.35 * avg_energy
        - 0.03 * avg_wall_hits
        - 0.01 * avg_no_progress
    )
    fitness = base_fitness + prior_weight * prior_agreement

    if return_details:
        return {
            'fitness': float(fitness),
            'base_fitness': float(base_fitness),
            'prior_agreement': float(prior_agreement),
            'avg_reward': float(avg_reward),
            'avg_energy': float(avg_energy),
            'avg_foods': float(avg_foods),
            'avg_wall_hits': float(avg_wall_hits),
            'avg_no_progress': float(avg_no_progress),
            'avg_progress': float(avg_progress),
            'task_score': float(task_score),
        }

    return float(fitness)
