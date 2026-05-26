from pathlib import Path
import argparse
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd):
    print('\n$ ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_foods(run_dir):
    path = PROJECT_ROOT / run_dir / 'benchmark_summary.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return float(summary['neat_best']['foods']['mean'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0,1,2,3,4,5,42,100')
    p.add_argument('--generations', type=int, default=60)
    p.add_argument('--base-dir', default='runs/sweep')
    p.add_argument('--benchmark-seeds', type=int, default=100)
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    results = []

    for seed in seeds:
        run_dir = f'{args.base_dir}/seed_{seed}'
        run_cmd([
            sys.executable,
            'experiments/train_recurrent_neat.py',
            '--generations', str(args.generations),
            '--seed', str(seed),
            '--output-dir', run_dir,
            '--top-k', '10',
        ])
        run_cmd([
            sys.executable,
            'experiments/benchmark_best_recurrent.py',
            '--run-dir', run_dir,
            '--num-seeds', str(args.benchmark_seeds),
            '--seed', '0',
        ])
        foods = load_foods(run_dir)
        results.append({'seed': seed, 'run_dir': run_dir, 'foods': foods})
        print(f'\n[Sweep] seed={seed} foods={foods} run_dir={run_dir}')

    results = sorted(results, key=lambda r: r['foods'] if r['foods'] is not None else -1, reverse=True)
    out_path = PROJECT_ROOT / args.base_dir / 'sweep_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\nBest runs:')
    for row in results[:5]:
        print(row)
    print('\nSaved:', out_path)


if __name__ == '__main__':
    main()
