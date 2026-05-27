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
Gen-2 NEAT seeded from top-k         16.57     1.21        6.04         93.1
Greedy oracle                        17.80     0.00        5.50        100.0
```

### Student-prior weight sweep

The initial student-prior run used `prior_weight=1.0` and collapsed to
4.45 foods (25% oracle). To check whether the failure was the method
or the hyperparameter, the prior weight was swept for both student
types (40 generations, training seed 123).

`StudentPrior` was also extended to accept the recurrent student
checkpoint, and now resets the student's hidden state at the start of
each genome-evaluation episode (the recurrent prior would otherwise
leak hidden state across the 5 evaluation episodes, the same class of
bug fixed in the benchmark and teacher collector).

Sparse benchmark over 100 seeds (10000..10099):

```text
prior_weight    Feedforward prior         Recurrent prior
                foods   walls             foods   walls
0.10             2.30   11.14              5.65    8.50
0.25            13.67    0.67             12.39    2.69
0.50             8.67    3.73             14.28    2.57
1.00             4.45    6.80               -       -
```

Best of each:

```text
Feedforward prior @ 0.25: foods=13.67  walls=0.67   76.8% of oracle
Recurrent prior   @ 0.50: foods=14.28  walls=2.57   80.2% of oracle
```

### Combining seeded reseed with a tuned student prior

The previous interpretation said the prior caps gen-2 near the
student's ability. That holds only when the prior is the only
gen-2 mechanism. When the prior is combined with seeded reseed
(population initialised from gen-1 top-k genomes), the prior stops
being a ceiling and becomes a refinement signal on top of an
already-strong starting point.

`train_recurrent_neat_seeded.py` was extended to accept an optional
`--student-path` and `--prior-weight`. With those set, fitness uses
the same student-prior shaping as the standalone student-prior
training; without them, it falls back to plain
`evaluate_recurrent_genome`.

Three training seeds (123, 200, 300), 40 generations each.
Sparse benchmark over 100 seeds (10000..10099):

```text
                            seed=123   seed=200   seed=300   mean (valid)
seeded only                 16.57      16.85      17.23      16.88  (n=3)
seeded + FF prior @0.25     CRASH*     17.19      17.61      17.40  (n=2)
seeded + Recurrent @0.5     17.00      17.21      17.73      17.31  (n=3)

mean wall hits per episode:
seeded only                  1.21       0.68       0.42       0.77
seeded + FF prior @0.25      -          0.31       0.05       0.18
seeded + Recurrent @0.5      0.43       0.18       0.02       0.21
```

*The `seeded + feedforward prior` run at seed 123 hit an
`AssertionError` deep inside neat-python's genome mutation
(`assert new_id not in node_dict`). The same combination at
seeds 200 and 300 ran cleanly, and the recurrent-prior version at
seed 123 ran cleanly, so the crash is a seed-dependent edge case in
neat-python under our reseeded initial population. We did not try to
patch the library; the two surviving runs are reported.

Combining is robust:

- Both combined strategies beat seeded-only at **every** seed.
- Mean improvement is roughly +0.4 foods/episode (about 1.8 std
  errors above seeded-only with n=3), and wall hits drop
  substantially.
- `seeded + recurrent prior` at seed 300 reaches 17.73 foods,
  within 0.07 of the greedy oracle's 17.80.

### Interpretation

The student-prior mechanism does add value when combined with seeded
reseed:

```text
Gen-1 NEAT teacher                    16.10  foods   ( 90.4% oracle)
Recurrent student (distilled)         15.60  foods   ( 87.6% oracle)
Seeded reseed only (mean)             16.88  foods   ( 94.8% oracle)
Seeded + recurrent prior @0.5 (mean)  17.31  foods   ( 97.2% oracle)
Greedy oracle                         17.80  foods   (100.0% oracle)
```

Reading across all gen-2 results so far:

- Used alone, a student prior **caps** evolution near the student's
  ability, because agreement with a weaker policy outvotes task
  reward when the genome is also weak.
- Used as a refinement on top of seeded reseed, the prior **shapes**
  evolution: starting genomes already collect food well, and the
  agreement bonus pulls them toward the cleaner action patterns the
  student learned (sharply lower wall hits in particular).

Working interpretation:

```text
The distilled student is most useful as a behavioural refinement
signal on top of an already-strong evolutionary starting point,
not as the sole driver of gen-2 evolution.
```

This is the first gen-2 evolve → distill → evolve configuration that
clearly beats both gen-1 and either of its constituent strategies
(seeded-only or prior-only) in a multi-seed comparison.

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
- The multi-seed gen-2 comparison used n=3 training seeds; the
  combined-strategy improvement is consistent but only ~1.8 SE above
  seeded-only.
- One configuration (seeded + feedforward prior @0.25 at seed 123)
  hit a seed-dependent assertion inside neat-python and is excluded
  from that cell's mean.
- Sparse evaluation should be expanded to different grid sizes and
  perturbed environments.

## Next steps

```text
1. Run a full second distillation (gen-2 -> student-2) and check
   whether the loop is self-improving across iterations
2. Widen the gen-2 multi-seed comparison (n>=10) to tighten the
   improvement-vs-seed-noise estimate
3. Sweep prior_weight again at the new (seeded + prior) operating
   point - the previous sweep was prior-only and may not transfer
4. Add environment variation: grid sizes, obstacles, perturbations
5. Track performance per parameter across teacher and students
6. Investigate the seed=123 + feedforward-prior + seeded-reseed
   neat-python assertion (innovation tracker collision)
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
