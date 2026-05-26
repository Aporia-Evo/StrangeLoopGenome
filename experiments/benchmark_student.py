from pathlib import Path
import argparse
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from slg.agents.oracle import greedy_action, random_action
from slg.distill.student import load_student
from slg.envs.gridworld import GridWorld
from slg.utils.reproducibility import set_global_seed


def run_episode(policy_fn, seed, size=8, max_steps=96):
    env = GridWorld(size=size, max_steps=max_steps, seed=seed)
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
    sparse_score = foods - 0.05 * wall_hits

    return {
        'seed': seed,
        'foods': foods,
        'wall_hits': wall_hits,
        'energy': float(info['energy']),
        'steps': steps,
        'steps_per_food': steps / max(1, foods),
        'sparse_score': sparse_score,
        'shaped_reward': shaped_reward,
        'total_progress': float(info.get('total_progress', 0.0)),
    }


def summarize(rows):
    keys = [
        'foods', 'wall_hits', 'energy', 'steps', 'steps_per_food',
        'sparse_score', 'shaped_reward', 'total_progress'
    ]
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return summary


def benchmark(args):
    set_global_seed(args.seed)
    run_path = PROJECT_ROOT / args.run_dir
    student_path = run_path / args.student_name
    model, _, _ = load_student(student_path)
    rng = np.random.default_rng(args.seed)

    policies = {
        'student': lambda obs: model.act(obs),
        'greedy_oracle': greedy_action,
        'random': lambda obs: random_action(rng),
    }

    all_rows = []
    summaries = {}
    for policy_name, policy_fn in policies.items():
        rows = []
        for seed in range(args.num_seeds):
            row = run_episode(policy_fn, seed=seed)
            row['policy'] = policy_name
            rows.append(row)
            all_rows.append(row)
        summaries[policy_name] = summarize(rows)

    csv_path = run_path / 'student_benchmark_sparse.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = run_path / 'student_benchmark_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

    print('\nStudent sparse benchmark over', args.num_seeds, 'seeds')
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
    parser.add_argument('--run-dir', type=str, default='runs/latest')
    parser.add_argument('--student-name', type=str, default='student_policy.pt')
    parser.add_argument('--num-seeds', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    benchmark(parse_args())
