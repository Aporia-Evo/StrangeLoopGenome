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

A full `evolve -> distill -> evolve -> distill` loop on an 8x8
GridWorld reached oracle-level performance at the training grid
size, but a cross-size benchmark exposed that the iterated
champion is brittle. Mean foods per episode:

```text
Policy                size=8 (train)   worst across 6,8,10,12,16
greedy_oracle              17.80              17.66
neat_teacher_1             16.10              12.97
student_1_recurrent        15.60              14.82
neat_teacher_2             17.73               0.82  *
student_2_feedforward      17.80               1.68  *
student_2_recurrent        16.93              15.55
random                      0.50               0.15
```

`*` Collapses out-of-distribution: the Milestone-4 "matches the
oracle" headline holds only on the 8x8 training grid; both the
iteration-2 NEAT teacher and the iteration-2 feedforward student
fall apart on grid sizes >= 10.

The actually robust learned policies are the **recurrent** students.
Student-2 recurrent (worst case 15.55, mean 16.40 across sizes) is
the best learned generalist; student-1 recurrent actually *improves*
with grid size. The recurrent capacity carries the policy across
distributional shifts that the feedforward student and the NEAT
genomes cannot survive.

The task is intentionally simple and has a known optimum; this is
a PoC, not a generality claim. See RESULTS.md for the full results,
the cross-size table, and reproducibility notes.

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
    matching the greedy oracle at the training grid size
[x] Cross-grid-size benchmark on sizes 6, 8, 10, 12, 16:
    iteration-2 champion is brittle, recurrent students
    generalise; student-2 recurrent is the best generalist
    (worst case 15.55 foods, mean 16.40)
[x] Energy-weight ablation (3 seeds x {0, 0.05, 0.15, 0.35}):
    the original 0.35 default was over-tuned. Weight 0.15
    gives the best mean (12.30 vs 9.56 foods) and the single
    best run (17.74). Removing the energy term entirely
    collapses two of three seeds - energy is doing real
    search-shaping work, but through a crude coupling.
[x] Lightweight energy-as-dynamics attempt (inner_steps and
    convergence_weight) - negative result. Naive inner
    iterations make things worse, and the convergence metric
    is gameable by dead networks. A proper Hopfield-style
    energy function and architecture are needed.
[ ] Design and implement a proper energy-based network class
    with explicit E(x, obs) and provably convergent dynamics
[ ] Add obstacles and stochastic perturbations to GridWorld
[ ] Train with environment variation, then re-benchmark
[ ] Re-run earlier milestones with energy_weight=0.15
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
