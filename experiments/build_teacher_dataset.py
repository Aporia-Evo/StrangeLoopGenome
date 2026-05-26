from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.distill.teacher_dataset import build_teacher_dataset
from slg.utils.reproducibility import set_global_seed


def load_config():
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )


def resolve_run_path(run_dir):
    run_path = PROJECT_ROOT / run_dir
    top_genomes_path = run_path / 'top_genomes.pkl'

    if top_genomes_path.exists():
        return run_path

    runs_root = PROJECT_ROOT / 'runs'
    available = []
    if runs_root.exists():
        available = sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in runs_root.glob('*/top_genomes.pkl')
        )

    hint = '\n'.join(f'  - {item}' for item in available) or '  none found'

    raise FileNotFoundError(
        f'Missing {top_genomes_path}\n\n'
        'Build the teacher dataset from a run directory that contains top_genomes.pkl.\n'
        'Examples:\n'
        '  python experiments/build_teacher_dataset.py --run-dir runs/latest\n'
        '  python experiments/train_recurrent_neat.py --output-dir runs/seed_42\n'
        '  python experiments/build_teacher_dataset.py --run-dir runs/seed_42\n\n'
        f'Available top_genomes files:\n{hint}'
    )


def run(args):
    set_global_seed(args.seed)
    run_path = resolve_run_path(args.run_dir)
    config = load_config()

    summary = build_teacher_dataset(
        top_genomes_path=run_path / 'top_genomes.pkl',
        config=config,
        output_path=run_path / args.output_name,
        num_genomes=args.num_genomes,
        seeds_per_genome=args.seeds_per_genome,
        seed_offset=args.seed,
    )

    summary_path = run_path / 'teacher_dataset_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\nTeacher dataset built')
    for key, value in summary.items():
        print(f'{key}: {value}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default='runs/latest')
    parser.add_argument('--output-name', type=str, default='teacher_dataset.npz')
    parser.add_argument('--num-genomes', type=int, default=5)
    parser.add_argument('--seeds-per-genome', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
