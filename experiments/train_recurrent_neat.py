from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.archive import ArchiveReporter, GenomeArchive
from slg.evolution.eval_recurrent_genome import evaluate_recurrent_genome
from slg.utils.reproducibility import save_run_config, set_global_seed


energy_weight = 0.35


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        details = evaluate_recurrent_genome(
            genome,
            config,
            return_details=True,
            energy_weight=energy_weight,
        )
        genome.fitness = details['fitness']
        genome.slg_metrics = details


def run(generations=40, seed=1, output_dir='runs/latest', top_k=10, energy_weight_value=0.35):
    global energy_weight
    energy_weight = energy_weight_value

    set_global_seed(seed)

    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'
    output_path = PROJECT_ROOT / output_dir

    save_run_config(
        output_path,
        {
            'script': 'train_recurrent_neat.py',
            'generations': generations,
            'seed': seed,
            'output_dir': str(output_path),
            'top_k': top_k,
            'energy_weight': energy_weight,
            'config_path': str(config_path),
        },
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    archive = GenomeArchive(output_dir=output_path, top_k=top_k)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(ArchiveReporter(archive))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(eval_genomes, generations)

    summary = archive.save()

    print(f'\nArchive summary saved to {output_path}/')
    print(summary)

    return archive.best_genome


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='runs/latest')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--energy-weight', type=float, default=0.35)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(
        generations=args.generations,
        seed=args.seed,
        output_dir=args.output_dir,
        top_k=args.top_k,
        energy_weight_value=args.energy_weight,
    )
