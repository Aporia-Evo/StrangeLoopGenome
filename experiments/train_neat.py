from pathlib import Path

import neat

from slg.evolution.eval_genome import evaluate_genome


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)


def run():
    config_path = (
        Path(__file__).resolve().parent.parent
        / 'slg'
        / 'evolution'
        / 'config-feedforward'
    )

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

    winner = population.run(eval_genomes, 20)

    print('\nBest genome:\n')
    print(winner)


if __name__ == '__main__':
    run()
