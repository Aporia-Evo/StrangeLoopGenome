from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat
import numpy as np

from slg.envs.gridworld import GridWorld
from slg.evolution.eval_recurrent_genome import evaluate_recurrent_genome


def load_config():
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )


def render_grid(env):
    grid = [['.' for _ in range(env.size)] for _ in range(env.size)]
    ax, ay = env.agent_pos
    fx, fy = env.food_pos
    grid[fy][fx] = 'F'
    grid[ay][ax] = 'A'
    return '\n'.join(' '.join(row) for row in grid)


def replay_episode(genome, config, seed=0, render=False):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    env = GridWorld(size=8, max_steps=96, seed=seed)

    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    if render:
        print('\nInitial state')
        print(render_grid(env))

    while not done:
        output = net.activate(obs)
        action = int(np.argmax(output))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            print(f'\nstep={steps} action={action} reward={reward:.3f}')
            print(render_grid(env))

    return {
        'seed': seed,
        'reward': float(total_reward),
        'steps': steps,
        'foods': int(info['foods_collected']),
        'wall_hits': int(info['wall_hits']),
        'energy': float(info['energy']),
    }


def main():
    config = load_config()
    genome_path = PROJECT_ROOT / 'runs' / 'latest' / 'best_genome.pkl'

    if not genome_path.exists():
        raise FileNotFoundError(
            'Missing runs/latest/best_genome.pkl. Run train_recurrent_neat.py first.'
        )

    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    details = evaluate_recurrent_genome(
        genome,
        config,
        episodes=25,
        return_details=True,
    )

    print('\nEvaluation over 25 fixed episodes')
    for key, value in details.items():
        print(f'{key}: {value}')

    print('\nReplay summaries')
    summaries = [replay_episode(genome, config, seed=i) for i in range(5)]
    for summary in summaries:
        print(summary)

    print('\nRendered episode for seed 0')
    replay_episode(genome, config, seed=0, render=True)


if __name__ == '__main__':
    main()
