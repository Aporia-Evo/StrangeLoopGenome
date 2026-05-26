# Results

## Milestone 1: Energy-shaped recurrent NEAT on GridWorld

This milestone tests whether a compact evolved recurrent network can learn a robust GridWorld food-collection policy when evolution is guided by local energy/progress signals.

The result is not intended as a claim of general intelligence. It is a first PoC that shows the search becomes much more effective when local progress is made visible to evolution.

## Setup

Environment:

```text
GridWorld size: 8 x 8
Episode length: 96 steps
Task: collect as much food as possible
Actions: up, down, left, right, wait
Inputs: delta_x, delta_y, energy, agent_x, agent_y
```

Evolution:

```text
Algorithm: recurrent NEAT
Population: 80
Generations: 40
Archive: Top-K genomes
Training seeds: fixed small seed set during evaluation
```

Training signal:

```text
fitness = task_score
        - energy penalty
        - wall-hit penalty
        - stagnation penalty
```

The important shaping term is local progress toward food:

```text
progress = previous_distance_to_food - current_distance_to_food
```

This made the local search landscape smoother and reduced the tendency to fall into useless oscillatory attractors.

## Sparse benchmark

After training, the best archived genome was evaluated over 100 seeds and compared against:

- a greedy oracle
- a random agent

Benchmark command:

```bash
python experiments/benchmark_best_recurrent.py
```

Results:

```text
Policy          Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
NEAT best       15.18     0.62        6.43         15.15          17.20
Greedy oracle   17.95     0.00        5.45         17.95          20.45
Random          0.49      7.99        85.23        0.09           -2.00
```

The best evolved agent reached approximately:

```text
15.18 / 17.95 = 84.6% of greedy oracle food collection
```

with a compact topology:

```text
nodes:               9
enabled connections: 10
```

## Interpretation

Initial runs without local progress shaping were slow and unstable. The networks often discovered partial strategies, collided with walls, or fell into oscillatory local attractors near the target.

After adding progress shaping and stagnation penalties, evolution quickly discovered a compact and robust policy.

Working interpretation:

```text
Search alone is not enough.
Evolution becomes much more effective when the environment exposes locally meaningful energy/progress gradients.
```

This supports the project hypothesis that compact intelligence-like dynamics may be cultivated through:

```text
evolution + local energy field + archive + distillation
```

rather than brute scale alone.

## Current limitations

- The task is very simple.
- Progress shaping is a strong inductive bias.
- The benchmark currently uses one environment type only.
- The recurrent genome is not yet distilled.
- No evolve-distill-evolve cycle has been run yet.
- Sparse evaluation should be expanded to different grid sizes and perturbed environments.

## Next steps

```text
1. Make benchmark runs reproducible
2. Use Top-K genomes as teachers
3. Build distillation
4. Distill compact policy/student networks
5. Re-seed evolution from distilled students
6. Run evolve → distill → evolve
7. Compare against the original NEAT run
```

## Files produced by current benchmark

```text
runs/latest/best_genome.pkl
runs/latest/top_genomes.pkl
runs/latest/archive_summary.json
runs/latest/benchmark_sparse.csv
runs/latest/benchmark_summary.json
```

These artifacts are local run outputs and are not committed by default.
