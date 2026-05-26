from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.archive import GenomeArchive
from slg.evolution.eval_recurrent_genome import evaluate_recurrent_genome


archive = GenomeArchive(output_dir='runs/latest', top_k=10)
current_generation = 0


def eval_genomes(genomes, config):
    global current_generation

    evaluated_population = {}

    for genome_id, genome in genomes:
        genome.fitness = evaluate_recurrent_genome(genome, config)
        evaluated_population[genome_id] = genome

    best = archive.update(current_generation, evaluated_population)

    if best is not None:
        print(
            f"\n[Archive] Best so far | "
            f"gen={best.generation} | "
            f"fitness={best.fitness:.3f} | "
            f"nodes={best.nodes} | "
            f"connections={best.connections}"
        )


def run(generations=40):
    global current_generation

    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    for generation in range(generations):
        current_generation = generation
        population.run(eval_genomes, 1)

    summary = archive.save()

    print('\nArchive summary saved to runs/latest/')
    print(summary)

    return archive.best_genome


if __name__ == '__main__':
    run()
