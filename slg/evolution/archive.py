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
        ranked = sorted(
            population.items(),
            key=lambda item: item[1].fitness if item[1].fitness is not None else -1e18,
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
            self.records.append(record)
            self.genomes.append((record, genome))

            if self.best_record is None or record.fitness > self.best_record.fitness:
                self.best_record = record
                self.best_genome = genome

        self.genomes = sorted(
            self.genomes,
            key=lambda item: item[0].fitness,
            reverse=True,
        )[: self.top_k]

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
