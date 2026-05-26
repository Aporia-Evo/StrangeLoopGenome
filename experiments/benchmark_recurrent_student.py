from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from slg.agents.oracle import greedy_action, random_action
from slg.distill.recurrent_student import load_recurrent_student
from slg.envs.gridworld import GridWorld


def run_episode(policy, seed):
    env = GridWorld(seed=seed)
    obs = env.reset()
    done = False
    shaped = 0.0
    steps = 0
    if hasattr(policy, 'reset'):
        policy.reset()
    while not done:
        action = policy.act(obs) if hasattr(policy, 'act') else policy(obs)
        obs, reward, done, info = env.step(action)
        shaped += reward
        steps += 1
    foods = int(info['foods_collected'])
    walls = int(info['wall_hits'])
    return {'foods': foods, 'wall_hits': walls, 'steps_per_food': steps / max(1, foods), 'sparse_score': foods - 0.05 * walls, 'shaped_reward': shaped}


def summarize(rows):
    keys = rows[0].keys()
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='runs/latest')
    p.add_argument('--student-name', default='recurrent_student.pt')
    p.add_argument('--num-seeds', type=int, default=100)
    args = p.parse_args()
    run_path = PROJECT_ROOT / args.run_dir
    student, _, _ = load_recurrent_student(run_path / args.student_name)
    rng = np.random.default_rng(0)
    policies = {'recurrent_student': student, 'greedy_oracle': greedy_action, 'random': lambda obs: random_action(rng)}
    print('\nRecurrent student sparse benchmark')
    for name, policy in policies.items():
        rows = [run_episode(policy, s) for s in range(args.num_seeds)]
        s = summarize(rows)
        print(f'\nPolicy: {name}')
        for k, (m, sd) in s.items():
            print(f'  {k}: {m:.3f} ± {sd:.3f}')


if __name__ == '__main__':
    main()
