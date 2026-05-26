from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.archive import ArchiveReporter, GenomeArchive
from slg.evolution.eval_recurrent_with_student_prior import (
    StudentPrior,
    evaluate_recurrent_genome_with_student_prior,
)
from slg.utils.reproducibility import save_run_config, set_global_seed


student_prior = None
prior_weight = 1.0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        details = evaluate_recurrent_genome_with_student_prior(
            genome,
            config,
            student_prior=student_prior,
            prior_weight=prior_weight,
            return_details=True,
        )
        genome.fitness = details['fitness']
        genome.slg_metrics = details


def main():
    global student_prior, prior_weight

    parser = argparse.ArgumentParser()
    parser.add_argument('--student-path', required=True)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output-dir', default='runs/evolve2')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--prior-weight', type=float, default=1.0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    prior_weight = args.prior_weight
    student_prior = StudentPrior(PROJECT_ROOT / args.student_path)

    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'
    output_path = PROJECT_ROOT / args.output_dir

    save_run_config(
        output_path,
        {
            'script': 'train_recurrent_neat_with_student_prior.py',
            'student_path': str(PROJECT_ROOT / args.student_path),
            'generations': args.generations,
            'seed': args.seed,
            'output_dir': str(output_path),
            'top_k': args.top_k,
            'prior_weight': args.prior_weight,
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

    archive = GenomeArchive(output_dir=output_path, top_k=args.top_k)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(ArchiveReporter(archive))
    population.add_reporter(neat.StatisticsReporter())

    population.run(eval_genomes, args.generations)
    summary = archive.save()

    print(f'\nSecond-generation archive saved to {output_path}/')
    print(summary)


if __name__ == '__main__':
    main()
