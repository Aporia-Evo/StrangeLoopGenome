from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat

from slg.evolution.archive import ArchiveReporter, GenomeArchive
from slg.evolution.eval_recurrent_genome import evaluate_recurrent_genome


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        details = evaluate_recurrent_genome(
            genome,
            config,
            return_details=True,
        )
        genome.fitness = details['fitness']
        genome.slg_metrics = details


def run(generations=40):
    config_path = PROJECT_ROOT / 'slg' / 'evolution' / 'config-recurrent'

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    archive = GenomeArchive(output_dir='runs/latest', top_k=10)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(ArchiveReporter(archive))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(eval_genomes, generations)

    summary = archive.save()

    print('\nArchive summary saved to runs/latest/')
    print(summary)

    return archive.best_genome


if __name__ == '__main__':
    run()
