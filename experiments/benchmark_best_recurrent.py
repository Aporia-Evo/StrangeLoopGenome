from pathlib import Path
import argparse
import csv
import json
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat
import numpy as np

from slg.agents.oracle import greedy_action, random_action
from slg.envs.gridworld import GridWorld
from slg.utils.reproducibility import save_run_config, set_global_seed


def load_config():
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )


def load_best_genome(run_dir):
    genome_path = Path(run_dir) / 'best_genome.pkl'

    if not genome_path.exists():
        raise FileNotFoundError(
            f'Missing {genome_path}. Run train_recurrent_neat.py first.'
        )

    with open(genome_path, 'rb') as f:
        return pickle.load(f)


def run_episode(policy_fn, seed, size=8, max_steps=96):
    env = GridWorld(size=size, max_steps=max_steps, seed=seed)
    if hasattr(policy_fn, 'reset'):
        policy_fn.reset()
    obs = env.reset()
    done = False
    steps = 0

    shaped_reward = 0.0

    while not done:
        action = int(policy_fn(obs))
        obs, reward, done, info = env.step(action)
        shaped_reward += reward
        steps += 1

    foods = int(info['foods_collected'])
    wall_hits = int(info['wall_hits'])
    energy = float(info['energy'])
    total_progress = float(info.get('total_progress', 0.0))

    sparse_score = foods - 0.05 * wall_hits
    steps_per_food = steps / max(1, foods)

    return {
        'seed': seed,
        'foods': foods,
        'wall_hits': wall_hits,
        'energy': energy,
        'steps': steps,
        'steps_per_food': steps_per_food,
        'sparse_score': sparse_score,
        'shaped_reward': shaped_reward,
        'total_progress': total_progress,
    }


def summarize(rows):
    keys = [
        'foods',
        'wall_hits',
        'energy',
        'steps',
        'steps_per_food',
        'sparse_score',
        'shaped_reward',
        'total_progress',
    ]

    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
        }

    return summary


def make_neat_policy(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    def policy(obs):
        output = net.activate(obs)
        return int(np.argmax(output))

    policy.reset = net.reset
    return policy


def benchmark(num_seeds=100, seed=0, run_dir='runs/latest', first_seed=10000):
    set_global_seed(seed)

    run_path = PROJECT_ROOT / run_dir
    config = load_config()
    genome = load_best_genome(run_path)

    policies = {
        'neat_best': make_neat_policy(genome, config),
        'greedy_oracle': greedy_action,
        'random': lambda obs, rng=np.random.default_rng(seed): random_action(rng),
    }

    run_path.mkdir(parents=True, exist_ok=True)

    save_run_config(
        run_path,
        {
            'script': 'benchmark_best_recurrent.py',
            'num_seeds': num_seeds,
            'seed': seed,
            'first_seed': first_seed,
            'run_dir': str(run_path),
        },
    )

    all_rows = []
    summaries = {}

    for policy_name, policy_fn in policies.items():
        rows = []
        for episode_index in range(num_seeds):
            episode_seed = first_seed + episode_index
            row = run_episode(policy_fn, seed=episode_seed)
            row['policy'] = policy_name
            rows.append(row)
            all_rows.append(row)

        summaries[policy_name] = summarize(rows)

    csv_path = run_path / 'benchmark_sparse.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'policy',
                'seed',
                'foods',
                'wall_hits',
                'energy',
                'steps',
                'steps_per_food',
                'sparse_score',
                'shaped_reward',
                'total_progress',
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = run_path / 'benchmark_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

    print('\nSparse benchmark over', num_seeds, 'seeds')
    for policy_name, summary in summaries.items():
        print(f'\nPolicy: {policy_name}')
        print(f"  foods:          {summary['foods']['mean']:.3f} ± {summary['foods']['std']:.3f}")
        print(f"  wall_hits:      {summary['wall_hits']['mean']:.3f} ± {summary['wall_hits']['std']:.3f}")
        print(f"  steps_per_food: {summary['steps_per_food']['mean']:.3f} ± {summary['steps_per_food']['std']:.3f}")
        print(f"  sparse_score:   {summary['sparse_score']['mean']:.3f} ± {summary['sparse_score']['std']:.3f}")
        print(f"  shaped_reward:  {summary['shaped_reward']['mean']:.3f} ± {summary['shaped_reward']['std']:.3f}")

    print('\nSaved:')
    print(csv_path)
    print(json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-dir', type=str, default='runs/latest')
    parser.add_argument('--first-seed', type=int, default=10000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark(
        num_seeds=args.num_seeds,
        seed=args.seed,
        run_dir=args.run_dir,
        first_seed=args.first_seed,
    )
