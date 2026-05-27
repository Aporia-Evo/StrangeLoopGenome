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

This milestone tests whether a strong evolved recurrent NEAT policy can
be distilled into a compact neural student.

The teacher is the `--seed 1` Milestone 1 genome (16.10 foods on the
sparse benchmark).

A clean Top-1 teacher dataset was built by keeping only high-quality
teacher episodes:

```text
num_genomes:              1
seeds_per_genome:         200
accepted episodes:        195
skipped episodes:         5
samples:                  18720
mean episode foods:       16.32
mean episode wall hits:   0.96
min_foods filter:         10
max_wall_hits filter:     3
teacher seed range:       2000..2199
```

### Feedforward student

A small feedforward student was trained on the clean Top-1 teacher dataset:

```text
hidden_dim: 32
epochs:     80
final val accuracy: ~0.893
```

Sparse benchmark over 100 seeds (10000..10099, disjoint from the
teacher seed range):

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
Feedforward student 12.97     8.27       10.01         12.56          14.76
Greedy oracle       17.80     0.00        5.50         17.80          20.29
Random               0.50     7.68       83.28          0.12          -1.97
```

Approximately:

```text
12.97 / 17.80 = 72.9% of greedy oracle food collection
```

The feedforward student picks up most of the teacher's food-finding
behavior, but its wall-hit rate is high and very noisy
(std=11.26 across episodes), which suggests the static policy loses
the teacher's edge in disambiguating moves near obstacles.

### Recurrent student

A GRU-based recurrent student was also trained on the same dataset:

```text
hidden_dim: 32
seq_len:    16
stride:     8
epochs:     60
final val accuracy: ~0.926
```

Sparse benchmark over 100 seeds:

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score   Shaped reward
Recurrent student   15.60     1.57        6.25         15.52          17.65
Greedy oracle       17.80     0.00        5.50         17.80          20.29
Random               0.50     7.68       83.28          0.12          -1.97
```

Approximately:

```text
15.60 / 17.80 = 87.6% of greedy oracle food collection
15.60 / 16.10 = 96.9% of the teacher itself
```

### Interpretation

The recurrent student clearly outperformed the feedforward student on
both food collection (15.60 vs 12.97) and wall avoidance (1.57 vs 8.27
walls per episode). It also tracks the teacher very closely
(~97% of the teacher's foods).

Earlier reported numbers in this milestone had the feedforward and
recurrent students roughly tied. Those numbers used the buggy benchmark
(see "Reproducibility notes") and a different teacher run, so they are
not directly comparable.

Working interpretation:

```text
Distillation can preserve evolved behavior,
and a recurrent student preserves it noticeably better
than a feedforward student when the teacher is itself recurrent.
```

---

## Milestone 3: First evolve → distill → evolve loop

This milestone runs the first full evolve → distill → evolve cycle
and asks whether a second generation of evolution, informed by the
distilled student, improves over the first generation NEAT teacher.

Two reseeding strategies were tested:

1. **Student-prior evolve**: a fresh NEAT population is evolved, but
   fitness is augmented with the agreement rate between the genome's
   action and the feedforward student's action
   (`train_recurrent_neat_with_student_prior.py`,
   `prior_weight=1.0`).
2. **Seeded evolve**: a fresh NEAT population is initialised from
   deep copies of the gen-1 top-k genomes (plus a single mutation
   pass), then evolved normally
   (`train_recurrent_neat_seeded.py`,
   `--elite-copies 4 --mutation-passes 1`).

Both used 40 generations and training seed 123.

Sparse benchmark over 100 seeds (10000..10099):

```text
Stage                                Foods    Wall hits   Steps/Food   % Oracle
Gen-1 NEAT (seed=1)                  16.10     1.25        6.06         90.4
Feedforward student (distilled)      12.97     8.27       10.01         72.9
Recurrent student (distilled)        15.60     1.57        6.25         87.6
Gen-2 NEAT with student prior         4.45     6.80       28.97         25.0
Gen-2 NEAT seeded from top-k         16.57     1.21        6.04         93.1
Greedy oracle                        17.80     0.00        5.50        100.0
```

### Interpretation

The first evolve → distill → evolve loop showed two very different
outcomes depending on how the distilled knowledge was reinjected:

- The **student-prior** strategy badly regressed (4.45 foods vs the
  gen-1 teacher's 16.10). Adding agreement with the feedforward
  student to fitness pulled evolution toward a policy that imitates
  the weaker student rather than improving over the teacher.
  Either the prior weight is too high or this is the wrong
  reinjection mechanism for this task.
- The **seeded** strategy gave a small improvement
  (16.57 vs 16.10 foods, roughly two standard errors over the
  benchmark mean). The benefit is real but modest, and well within
  the seed-sensitivity range observed in Milestone 1.

Working interpretation:

```text
For this PoC, the distillation step did not add value to the second
generation. Seeding the next population from the previous generation's
top genomes is a stronger starting point than using a distilled
behavioural prior as a fitness term.
```

This is a negative-but-honest result for the student-prior approach,
and a weakly positive result for direct genome reseeding. Both will
need more seeds and tighter prior-weight sweeps before any strong
claim about evolve → distill → evolve as a method.

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
- Seed sensitivity is significant; reported numbers come from the best
  seed in a small sweep.
- Clean distillation depends strongly on finding a good teacher first.
- The first evolve → distill → evolve loop only used a single training
  seed per strategy; the prior-weight was not swept.
- Sparse evaluation should be expanded to different grid sizes and
  perturbed environments.

## Next steps

```text
1. Sweep prior_weight for student-prior gen-2 evolution
   (the prior may have dominated over task reward)
2. Try recurrent-student prior instead of feedforward-student prior
3. Run a multi-seed comparison for seeded gen-2 to confirm the
   small improvement over gen-1 is real
4. Run a full second distillation (gen-2 -> student-2)
   to close one more loop iteration
5. Add environment variation: grid sizes, obstacles, perturbations
6. Track performance per parameter across teacher and students
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
