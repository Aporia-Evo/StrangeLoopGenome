from pathlib import Path
import argparse
import copy
import pickle
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.archive import ArchiveReporter, GenomeArchive
from slg.evolution.eval_recurrent_genome import evaluate_recurrent_genome
from slg.evolution.eval_recurrent_with_student_prior import (
    StudentPrior,
    evaluate_recurrent_genome_with_student_prior,
)
from slg.utils.reproducibility import save_run_config, set_global_seed


def load_config():
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    return config, config_path


def load_seed_genomes(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.append(item[1] if isinstance(item, tuple) and len(item) == 2 else item)
        return out
    return [obj]


def make_population(seed_genomes, config, pop_size, elite_copies, mutation_passes):
    pop = {}
    key = 1
    for source in seed_genomes[:elite_copies]:
        g = copy.deepcopy(source)
        g.key = key
        g.fitness = None
        pop[key] = g
        key += 1
        if key > pop_size:
            return pop
    while key <= pop_size:
        g = copy.deepcopy(random.choice(seed_genomes))
        g.key = key
        g.fitness = None
        for _ in range(mutation_passes):
            g.mutate(config.genome_config)
        pop[key] = g
        key += 1
    return pop


student_prior = None
prior_weight = 0.0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        if student_prior is not None:
            details = evaluate_recurrent_genome_with_student_prior(
                genome,
                config,
                student_prior=student_prior,
                prior_weight=prior_weight,
                return_details=True,
            )
        else:
            details = evaluate_recurrent_genome(genome, config, return_details=True)
        genome.fitness = details['fitness']
        genome.slg_metrics = details


def main():
    global student_prior, prior_weight

    p = argparse.ArgumentParser()
    p.add_argument('--seed-genomes', required=True)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--output-dir', default='runs/evolve2_seeded')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--elite-copies', type=int, default=4)
    p.add_argument('--mutation-passes', type=int, default=1)
    p.add_argument('--student-path', default=None)
    p.add_argument('--prior-weight', type=float, default=0.0)
    args = p.parse_args()

    set_global_seed(args.seed)
    config, config_path = load_config()
    seed_path = PROJECT_ROOT / args.seed_genomes
    output_path = PROJECT_ROOT / args.output_dir
    seeds = load_seed_genomes(seed_path)

    if args.student_path:
        student_prior = StudentPrior(PROJECT_ROOT / args.student_path)
        prior_weight = args.prior_weight

    save_run_config(output_path, {
        'script': 'train_recurrent_neat_seeded.py',
        'seed_genomes': str(seed_path),
        'num_seed_genomes': len(seeds),
        'generations': args.generations,
        'seed': args.seed,
        'output_dir': str(output_path),
        'top_k': args.top_k,
        'elite_copies': args.elite_copies,
        'mutation_passes': args.mutation_passes,
        'student_path': str(PROJECT_ROOT / args.student_path) if args.student_path else None,
        'prior_weight': args.prior_weight,
        'config_path': str(config_path),
    })

    archive = GenomeArchive(output_dir=output_path, top_k=args.top_k)
    population = neat.Population(config)
    population.population = make_population(seeds, config, config.pop_size, args.elite_copies, args.mutation_passes)
    population.species.speciate(config, population.population, population.generation)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(ArchiveReporter(archive))
    population.add_reporter(neat.StatisticsReporter())

    print('\nSeeded evolution initialized')
    print('source:', seed_path)
    print('seed genomes:', len(seeds))
    print('population size:', len(population.population))

    population.run(eval_genomes, args.generations)
    print(archive.save())


if __name__ == '__main__':
    main()
