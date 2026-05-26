import copy
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GenomeRecord:
    generation: int
    genome_id: int
    fitness: float
    nodes: int
    connections: int
    enabled_connections: int


class GenomeArchive:
    def __init__(self, output_dir='runs/latest', top_k=10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.records = []
        self.genomes = []
        self.best_record = None
        self.best_genome = None

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
            )
            genome_copy = copy.deepcopy(genome)

            self.records.append(record)
            self.genomes.append((record, genome_copy))

            if self.best_record is None or record.fitness > self.best_record.fitness:
                self.best_record = record
                self.best_genome = genome_copy

        self.genomes = sorted(
            self.genomes,
            key=lambda item: item[0].fitness,
            reverse=True,
        )[: self.top_k]

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
