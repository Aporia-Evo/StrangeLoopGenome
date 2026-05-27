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

## Headline result

A full `evolve -> distill -> evolve -> distill` loop reached
oracle-level performance on the GridWorld PoC in two iterations:

```text
Stage                              Foods   Wall hits   % Oracle
Teacher-1 (gen-1 NEAT)             16.10    1.25        90.4
Student-1 recurrent (distilled)    15.60    1.57        87.6
Teacher-2 (gen-2 NEAT)             17.73    0.02        99.6
Student-2 feedforward (distilled)  17.80    0.00       100.0
Greedy oracle                      17.80    0.00       100.0
Random                              0.50    7.68         2.8
```

Two ingredients made the second iteration work:

1. Gen-2 evolution initialised its population from gen-1 top-k
   genomes (`--elite-copies 4 --mutation-passes 1`) and used the
   recurrent student-1 as a behavioural prior in fitness
   (`--prior-weight 0.5`).
2. Teacher-2 was much cleaner than teacher-1 (200/200 vs 195/200
   episodes accepted, mean walls 0.015 vs 0.96), so the
   second-pass student saw a near-deterministic optimum.

The task is intentionally simple and has a known optimum; this is
a PoC, not a generality claim. See RESULTS.md for the full results
and reproducibility notes.

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
[x] Milestone 1: energy-shaped recurrent NEAT baseline
[x] Reproducible benchmark with fresh state per episode and
    disjoint train/eval seeds
[x] Top-K archive of evolved genomes
[x] Top-1 clean distillation into feedforward and recurrent students
[x] First evolve -> distill -> evolve loop
    (student-prior, seeded reseed)
[x] Sweep prior_weight and try recurrent-student prior
[x] Multi-seed comparison: seeded + tuned student prior
    consistently beats seeded-only (17.31 vs 16.88 foods mean,
    n=3 seeds, recurrent prior at weight 0.5)
[x] Second distillation pass (gen-2 -> student-2):
    feedforward student-2 reaches 17.80 foods / 0 walls,
    matching the greedy oracle. One full loop iteration:
    16.10 -> 17.80 foods (90.4% -> 100% oracle).
[ ] Wider multi-seed sweep (n>=10) to tighten the estimate
[ ] Environment variation: grid sizes, obstacles, perturbations
[ ] Third loop iteration (gen-3 from student-2 prior)
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
