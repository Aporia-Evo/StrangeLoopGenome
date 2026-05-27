# Results

## Milestone 1: Energy-shaped recurrent NEAT on GridWorld

This milestone tests whether a compact evolved recurrent network can learn a robust GridWorld food-collection policy when evolution is guided by local energy/progress signals.

The result is not intended as a claim of general intelligence. It is a first PoC that shows the search becomes much more effective when local progress is made visible to evolution.

### Setup

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
Generations: 40+
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

### Sparse benchmark

After training, the best archived genome was evaluated over 100 seeds
disjoint from the training seeds, and compared against:

- a greedy oracle
- a random agent

Benchmark command:

```bash
python experiments/benchmark_best_recurrent.py
```

The benchmark uses fresh recurrent state per episode and benchmark seeds
in the range 10000-10099 (disjoint from training and teacher-collection
seeds, see "Reproducibility notes" below).

Result on the `--seed 1` training run (60 generations):

```text
Policy          Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
NEAT best       16.10     1.25        6.06         16.04          18.27
Greedy oracle   17.80     0.00        5.50         17.80          20.29
Random           0.50     7.68       83.28          0.12          -1.97
```

The best evolved agent reached approximately:

```text
16.10 / 17.80 = 90.4% of greedy oracle food collection
```

with a compact topology.

### Seed sensitivity

A sweep over training seeds (60 generations each, same config, same
benchmark) shows that this PoC is highly seed-sensitive:

```text
Training seed   Benchmark foods
            1   16.10
            2    3.37
            0    3.21
           42    0.85
```

A single training run is therefore a noisy estimate of what the method
can produce. Reported numbers in this milestone come from the best
seed in a small sweep, not from a single run with an arbitrary default
seed.

### Interpretation

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

---

## Milestone 2: Top-1 clean distillation preserves most behavior

This milestone tests whether a strong evolved recurrent NEAT policy can be distilled into a compact neural student.

A seed sweep found a strong teacher run:

```text
best sweep run: runs/sweep/seed_1
teacher benchmark: ~14.51 foods over 100 seeds
```

A clean Top-1 teacher dataset was then built by keeping only high-quality teacher episodes:

```text
num_genomes:              1
seeds_per_genome:         200
accepted episodes:        157
skipped episodes:         43
samples:                  15072
mean episode foods:       16.21
mean episode wall hits:   0.36
min_foods filter:         10
max_wall_hits filter:     3
```

### Feedforward student

A small feedforward student was trained on the clean Top-1 teacher dataset:

```text
hidden_dim: 32
epochs:     80
final val accuracy: ~0.892
```

Sparse benchmark over 100 seeds:

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
Feedforward student 15.44     0.09        6.47         15.44          17.52
Greedy oracle       17.95     0.00        5.45         17.95          20.45
Random              0.49      7.99        85.23        0.09           -2.00
```

This is approximately:

```text
15.44 / 17.95 = 86.0% of greedy oracle food collection
```

### Recurrent student

A GRU-based recurrent student was also trained on the same clean Top-1 teacher dataset:

```text
hidden_dim: 32
seq_len:    16
stride:     8
epochs:     60
final val accuracy: ~0.908
```

Sparse benchmark over 100 seeds:

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
Recurrent student   15.39     3.03        6.41         15.24          17.44
Greedy oracle       17.95     0.00        5.45         17.95          20.45
Random              0.49      7.99        85.23        0.09           -2.00
```

### Interpretation

The feedforward student unexpectedly preserved most of the strong teacher behavior. Earlier mixed Top-K distillation produced a much weaker student, but Top-1 clean distillation produced a compact student with near-teacher performance.

Key lesson:

```text
Teacher quality matters more than teacher quantity.
```

A mixed Top-K dataset can blur strategies. A clean Top-1 dataset can preserve a coherent policy.

The recurrent student achieved slightly higher imitation accuracy, but worse wall-hit behavior than the feedforward student. This suggests that recurrent capacity alone is not automatically better; the training objective and rollout stability matter.

Working interpretation:

```text
Distillation can preserve evolved behavior,
but only when the teacher signal is coherent and high quality.
```

---

## Reproducibility notes

Earlier versions of the benchmark and teacher-collection code had two
bugs that have since been fixed:

- The recurrent NEAT network was created once outside the episode loop
  in `benchmark_best_recurrent.py` and `slg/distill/teacher_dataset.py`,
  so its hidden state persisted across all 100 benchmark episodes (and
  across all 50 teacher episodes per genome). The fix calls
  `net.reset()` before every episode, matching training-time evaluation.
- Teacher dataset seeds (0..num_genomes*seeds_per_genome) overlapped the
  benchmark seeds (0..num_seeds), so the distilled student was being
  evaluated on environments that appeared in its training data. Benchmarks
  now default to `--first-seed 10000`, so episode seeds are disjoint
  from the teacher dataset's seed range.

For a strong genome (the `--seed 1` run above), running the benchmark
with the buggy and fixed code produces nearly identical foods/episode,
so the headline numbers above are stable. For weaker genomes the bugs
could matter more.

## Current limitations

- The task is still very simple.
- Progress shaping is a strong inductive bias.
- The benchmark currently uses one environment type only.
- Seed sensitivity is significant.
- Clean distillation depends strongly on finding a good teacher first.
- The first evolve → distill → evolve reseeding loop has not yet been run.
- Sparse evaluation should be expanded to different grid sizes and perturbed environments.

## Next steps

```text
1. Use the distilled student as a seed or behavioral prior
2. Run evolve → distill → evolve
3. Compare second-generation evolution against first-generation NEAT
4. Add environment variation: grid sizes, obstacles, perturbations
5. Track performance per parameter across teacher and students
6. Study why feedforward student reduced wall hits better than recurrent student
```

## Files produced by current benchmark

```text
runs/latest/best_genome.pkl
runs/latest/top_genomes.pkl
runs/latest/archive_summary.json
runs/latest/benchmark_sparse.csv
runs/latest/benchmark_summary.json
runs/*/teacher_top1_clean.npz
runs/*/student_top1_clean.pt
runs/*/recurrent_student.pt
runs/*/student_benchmark_summary.json
```

These artifacts are local run outputs and are not committed by default.
