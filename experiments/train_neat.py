from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.eval_genome import evaluate_genome


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)


def run(generations=20):
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-feedforward'

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

    final_winner = population.run(eval_genomes, generations)
    best_overall = stats.best_genome()

    print('\nFinal generation winner:\n')
    print(final_winner)

    print('\nBest genome over all generations:\n')
    print(best_overall)

    return best_overall


if __name__ == '__main__':
    run()
