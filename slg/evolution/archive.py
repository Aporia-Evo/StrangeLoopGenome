import copy
import json
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path

import neat


@dataclass
class GenomeRecord:
    generation: int
    genome_id: int
    fitness: float
    nodes: int
    connections: int
    enabled_connections: int
    metrics: dict = field(default_factory=dict)


class GenomeArchive:
    def __init__(self, output_dir='runs/latest', top_k=10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.records = []
        self.genomes = []
        self.best_record = None
        self.best_genome = None

    def _signature(self, genome):
        enabled = tuple(
            sorted(
                key
                for key, conn in genome.connections.items()
                if conn.enabled
            )
        )
        return (tuple(sorted(genome.nodes.keys())), enabled)

    def update(self, generation, population):
        evaluated = {
            genome_id: genome
            for genome_id, genome in population.items()
            if genome.fitness is not None
        }

        if not evaluated:
            return None

        ranked = sorted(
            evaluated.items(),
            key=lambda item: item[1].fitness,
            reverse=True,
        )

        for genome_id, genome in ranked[: self.top_k]:
            record = GenomeRecord(
                generation=generation,
                genome_id=genome_id,
                fitness=float(genome.fitness),
                nodes=len(genome.nodes),
                connections=len(genome.connections),
                enabled_connections=sum(
                    1 for conn in genome.connections.values() if conn.enabled
                ),
                metrics=getattr(genome, 'slg_metrics', {}),
            )
            genome_copy = copy.deepcopy(genome)

            self.records.append(record)
            self.genomes.append((record, genome_copy))

            if self.best_record is None or record.fitness > self.best_record.fitness:
                self.best_record = record
                self.best_genome = genome_copy

        unique = {}
        for record, genome in sorted(
            self.genomes,
            key=lambda item: item[0].fitness,
            reverse=True,
        ):
            signature = self._signature(genome)
            if signature not in unique:
                unique[signature] = (record, genome)

            if len(unique) >= self.top_k:
                break

        self.genomes = list(unique.values())

        return self.best_record

    def save(self):
        if self.best_genome is not None:
            with open(self.output_dir / 'best_genome.pkl', 'wb') as f:
                pickle.dump(self.best_genome, f)

        with open(self.output_dir / 'top_genomes.pkl', 'wb') as f:
            pickle.dump(self.genomes, f)

        summary = {
            'best': asdict(self.best_record) if self.best_record else None,
            'top_k': [asdict(record) for record, _ in self.genomes],
        }

        with open(self.output_dir / 'archive_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        return summary


class ArchiveReporter(neat.reporting.BaseReporter):
    def __init__(self, archive):
        self.archive = archive
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        best = self.archive.update(self.generation, population)

        if best is not None:
            metrics = best.metrics or {}
            print(
                f"\n[Archive] Best so far | "
                f"gen={best.generation} | "
                f"fitness={best.fitness:.3f} | "
                f"foods={metrics.get('avg_foods', 0.0):.2f} | "
                f"reward={metrics.get('avg_reward', 0.0):.3f} | "
                f"energy={metrics.get('avg_energy', 0.0):.3f} | "
                f"nodes={best.nodes} | "
                f"enabled={best.enabled_connections}"
            )
