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
from slg.distill.recurrent_student import load_recurrent_student
from slg.distill.student import load_student
from slg.envs.gridworld import GridWorld


def load_neat_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'),
    )


def load_genome(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_neat_policy(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    def policy(obs):
        return int(np.argmax(net.activate(obs)))

    policy.reset = net.reset
    return policy


def make_ff_student_policy(student_path):
    model, _, _ = load_student(student_path)

    def policy(obs):
        return int(model.act(obs))

    return policy


def make_rec_student_policy(student_path):
    model, _, _ = load_recurrent_student(student_path)

    def policy(obs):
        return int(model.act(obs))

    policy.reset = model.reset
    return policy


def make_random_policy(seed):
    rng = np.random.default_rng(seed)

    def policy(obs):
        return int(random_action(rng))

    return policy


def run_episode(policy, seed, size, max_steps):
    env = GridWorld(size=size, max_steps=max_steps, seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, info = env.step(int(policy(obs)))
    return {
        'foods': int(info['foods_collected']),
        'wall_hits': int(info['wall_hits']),
        'energy': float(info['energy']),
    }


def benchmark_policy(make_policy_fn, sizes, num_seeds, first_seed):
    rows = []
    for size in sizes:
        max_steps = 12 * size
        policy = make_policy_fn()
        foods_list = []
        wall_list = []
        for i in range(num_seeds):
            r = run_episode(policy, first_seed + i, size, max_steps)
            foods_list.append(r['foods'])
            wall_list.append(r['wall_hits'])
        rows.append({
            'size': size,
            'max_steps': max_steps,
            'foods_mean': float(np.mean(foods_list)),
            'foods_std': float(np.std(foods_list)),
            'walls_mean': float(np.mean(wall_list)),
            'walls_std': float(np.std(wall_list)),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', default='6,8,10,12,16')
    p.add_argument('--num-seeds', type=int, default=100)
    p.add_argument('--first-seed', type=int, default=10000)
    p.add_argument('--random-seed', type=int, default=0)
    p.add_argument('--gen1-run', default='runs/latest_seed1')
    p.add_argument('--gen2-run', default='runs/m4_seeded_rec_s300')
    p.add_argument('--output', default='runs/cross_size_benchmark.json')
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(',')]
    neat_config = load_neat_config()
    gen1 = PROJECT_ROOT / args.gen1_run
    gen2 = PROJECT_ROOT / args.gen2_run

    policies = {
        'greedy_oracle':       lambda: greedy_action,
        'random':              lambda: make_random_policy(args.random_seed),
        'neat_teacher_1':      lambda: make_neat_policy(load_genome(gen1 / 'best_genome.pkl'), neat_config),
        'student_1_ff':        lambda: make_ff_student_policy(gen1 / 'student_top1_clean.pt'),
        'student_1_recurrent': lambda: make_rec_student_policy(gen1 / 'recurrent_student.pt'),
        'neat_teacher_2':      lambda: make_neat_policy(load_genome(gen2 / 'best_genome.pkl'), neat_config),
        'student_2_ff':        lambda: make_ff_student_policy(gen2 / 'student2_top1_clean.pt'),
        'student_2_recurrent': lambda: make_rec_student_policy(gen2 / 'recurrent_student2.pt'),
    }

    results = {}
    for name, make in policies.items():
        print(f'\n=== {name} ===')
        rows = benchmark_policy(make, sizes, args.num_seeds, args.first_seed)
        results[name] = rows
        for r in rows:
            print(f"  size={r['size']:2d}  foods={r['foods_mean']:6.2f} ± {r['foods_std']:.2f}  walls={r['walls_mean']:5.2f}")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

    csv_path = out_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['policy', 'size', 'max_steps', 'foods_mean', 'foods_std', 'walls_mean', 'walls_std'])
        for name, rows in results.items():
            for r in rows:
                writer.writerow([name, r['size'], r['max_steps'],
                                 f"{r['foods_mean']:.3f}", f"{r['foods_std']:.3f}",
                                 f"{r['walls_mean']:.3f}", f"{r['walls_std']:.3f}"])
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
