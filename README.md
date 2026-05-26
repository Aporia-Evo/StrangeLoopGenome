# StrangeLoopGenome

A small experimental framework for evolving compact, energy-guided, self-distilled neural systems.

## Core hypothesis

Intelligence-like behavior may emerge more efficiently from recursive cycles of:

```text
evolve → relax → distill → evolve
```

rather than from scale alone.

StrangeLoopGenome explores whether small evolving neural topologies can develop robust, compact, reusable dynamics when guided by:

- evolutionary topology search
- energy-based relaxation
- recursive distillation
- multi-scale structure
- adaptive fitness landscapes

A working framing from the first PoC:

```text
Intelligence does not emerge from search alone,
but from search over locally meaningful energy landscapes.
```

## Initial PoC

The first experiment uses a minimal GridWorld environment.

The agent receives local state information, acts in a small world, collects food, spends energy, and is penalized for inefficient dynamics such as wall collisions or stagnation.

We compare:

```text
A: random agent
B: greedy oracle
C: recurrent NEAT + energy/progress-shaped fitness
```

Main metric:

```text
performance per parameter
```

Secondary metrics:

- robustness on unseen seeds
- sparse task performance
- energy consumption
- network size
- wall collisions
- steps per food
- stability of internal dynamics

## Milestone 1 result

A recurrent NEAT agent evolved with energy/progress-shaped fitness reached roughly 85% of a greedy oracle baseline on a 100-seed sparse benchmark.

```text
Policy          Foods     Wall hits   Steps/Food   Sparse score
NEAT best       15.18     0.62        6.43         15.15
Greedy oracle   17.95     0.00        5.45         17.95
Random          0.49      7.99        85.23        0.09
```

The best evolved topology remained compact:

```text
nodes:               9
enabled connections: 10
benchmark seeds:     100
oracle performance:  ~84.6%
```

Interpretation:

```text
Blind evolution was slow and unstable.
Evolution over a locally meaningful progress/energy landscape became much more effective.
```

See [RESULTS.md](RESULTS.md) for details.

## Architecture

```text
Environment
    ↓
Population of genomes
    ↓
Energy/progress evaluation
    ↓
Fitness evaluation
    ↓
Selection + mutation
    ↓
Archive best genomes
    ↓
Distillation of elites
    ↓
New population seed
```

## Current roadmap

```text
1. Document Milestone 1 results in RESULTS.md
2. Make benchmark runs reproducible
3. Use Top-K genomes as teachers
4. Build distillation
5. Run evolve → distill → evolve
```

## Running the current PoC

Install dependencies:

```bash
pip install -r requirements.txt
```

Train recurrent NEAT:

```bash
python experiments/train_recurrent_neat.py
```

Evaluate the best archived genome:

```bash
python experiments/evaluate_best_recurrent.py
```

Run sparse benchmark against oracle and random baselines:

```bash
python experiments/benchmark_best_recurrent.py
```

Benchmark outputs are written to:

```text
runs/latest/benchmark_sparse.csv
runs/latest/benchmark_summary.json
```

## Project philosophy

StrangeLoopGenome treats intelligence not as a static model, but as a recursive developmental process.

The goal is not to build a large model.

The goal is to test whether compact intelligence-like dynamics can be cultivated through evolutionary pressure, energy landscapes, and repeated compression.
