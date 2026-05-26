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


def run(args):
    set_global_seed(args.seed)
    run_path = PROJECT_ROOT / args.run_dir
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
