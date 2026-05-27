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

## Milestone 4: Second distillation closes one full loop iteration

The strongest gen-2 NEAT run (seeded + recurrent prior at weight 0.5,
training seed 300, 17.73 foods on the sparse benchmark) was used as
the teacher for a second distillation pass. The pipeline now reads:

```text
gen-1 NEAT   -- distill -->  student-1
seeded + student-1 prior  -->  gen-2 NEAT
gen-2 NEAT  -- distill -->  student-2
```

### Teacher-2 dataset

```text
source genome:            runs/m4_seeded_rec_s300/top_genomes.pkl
num_genomes:              1
seeds_per_genome:         200
accepted episodes:        200
skipped episodes:         0
samples:                  19200
mean episode foods:       17.89
mean episode wall hits:   0.015
min_foods filter:         10
max_wall_hits filter:     3
teacher seed range:       3000..3199  (disjoint from teacher-1 range
                                       2000..2199 and benchmark range
                                       10000..10099)
```

Every single teacher-2 episode passed the filter (vs 195/200 for
teacher-1). The signal is much cleaner: almost no walls, near-oracle
food collection. Same `min_foods`/`max_wall_hits` thresholds.

### Feedforward student-2

```text
hidden_dim: 32
epochs:     80
final val accuracy: ~0.980
```

Sparse benchmark over 100 seeds:

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score
Feedforward stud-2  17.80     0.00        ~5.50        17.80
Greedy oracle       17.80     0.00        5.50         17.80
Random               0.50     7.68       83.28          0.12
```

The feedforward student-2 hits the same mean foods and the same
zero wall hits as the greedy oracle on this benchmark. A spot-check
shows the student matches the oracle's action on ~79% of in-trajectory
observations and ~61% on random observations - so it is not literally
re-implementing the oracle, but the alternative actions it picks reach
food in the same step count. Effectively oracle-equivalent on this
task.

### Recurrent student-2

```text
hidden_dim: 32
seq_len:    16
stride:     8
epochs:     60
final val accuracy: ~0.936
```

Sparse benchmark:

```text
Policy              Foods     Wall hits   Steps/Food   Sparse score
Recurrent stud-2   16.93     0.56        ~6.0         16.90
Greedy oracle      17.80     0.00         5.50        17.80
```

Roughly 95% of oracle - good but noticeably below feedforward
student-2. With a near-deterministic teacher, the recurrent capacity
does not pay off; the simpler feedforward student is the better fit
this iteration.

### One full loop in one table

```text
Stage                                    Foods   Wall hits   % Oracle
Teacher-1 (gen-1 NEAT, seed=1)           16.10    1.25         90.4
Student-1 feedforward                    12.97    8.27         72.9
Student-1 recurrent                      15.60    1.57         87.6
Teacher-2 (seeded + rec prior, s=300)    17.73    0.02         99.6
Student-2 feedforward                    17.80    0.00        100.0
Student-2 recurrent                      16.93    0.56         95.1
Greedy oracle                            17.80    0.00        100.0
Random                                    0.50    7.68          2.8
```

### Interpretation

One complete `evolve -> distill -> evolve -> distill` cycle moved the
best policy from 16.10 to 17.80 foods (90.4% -> 100.0% of oracle).
The loop is **self-improving** at this scale: every stage is
better than the corresponding stage in the previous iteration.

```text
Teacher-1   16.10  -->  Teacher-2   17.73
Student-1   15.60  -->  Student-2   17.80   (recurrent in iter 1, feedforward in iter 2)
```

The mechanism that drove the gain in iteration 2 was not the
distillation step in isolation: the gen-2 evolution itself improved
the teacher, and a much cleaner teacher then produced a much stronger
student. Cleanliness compounded - the teacher-2 dataset had 0% skipped
episodes vs 2.5% for teacher-1.

A few honest caveats:

- This is a single complete iteration. The next iteration may
  plateau (the policy is at oracle-level foods and zero walls,
  there is nothing left to learn on this benchmark) or regress
  (the student-2 may not be a useful prior for a gen-3 evolution).
- The task is simple and has a known optimum. The interesting
  question is whether the same pattern holds on harder tasks where
  the oracle is unknown.
- The recurrent student lost to the feedforward student in
  iteration 2. This is consistent with the task itself being
  essentially memoryless once the teacher is good - recurrent
  capacity helps when the teacher is noisy.

Working interpretation (subject to revision in Milestone 5):

```text
The evolve -> distill -> evolve -> distill loop is genuinely
self-improving on this PoC task at the training grid size, and
converges to the oracle in two iterations. The distillation step
compounds the gen-2 evolution gain into a cleaner student that
essentially matches the optimum.
```

Milestone 5 below tests this loop on grid sizes outside the
training distribution and finds that the in-distribution
self-improvement claim does not transfer.

---

## Milestone 5: Cross-grid-size generalisation reveals brittleness

All training and benchmarking up to Milestone 4 used an 8x8 grid.
This milestone benchmarks every policy on sizes 6, 8, 10, 12, 16
with `max_steps = 12 * size` (so the same per-cell time budget),
using the same 100-seed benchmark range (10000..10099) per size.

Sparse benchmark, mean foods per episode:

```text
Policy                size=6  size=8  size=10  size=12  size=16   mean   worst
greedy_oracle         17.85   17.80   17.79    17.66    17.80    17.78  17.66
neat_teacher_1        16.58   16.10   16.36    16.40    12.97    15.68  12.97
student_1_ff          17.85   12.97   11.25     9.45     2.78    10.86   2.78
student_1_recurrent   14.82   15.60   16.01    16.54    16.97    15.99  14.82
neat_teacher_2        17.75   17.73    1.11     0.89     0.82     7.66   0.82
student_2_ff          17.69   17.80   17.27    14.25     1.68    13.74   1.68
student_2_recurrent   15.60   16.93   17.25    16.67    15.55    16.40  15.55
random                 0.88    0.50    0.31     0.24     0.15     0.42   0.15
```

### What's actually happening

1. **The Milestone-4 champion is brittle.** NEAT teacher-2 collapses
   from 17.73 foods at size 8 to 1.11 at size 10 and 0.82 at
   size 16. It is a near-oracle on its training distribution and
   barely above random everywhere else.
2. **Student-2 feedforward inherits the brittleness with a delay.**
   It survives sizes 6-10 (>17 foods) but collapses at size 16
   (1.68 foods). Its closeness to the oracle on size 8 was a
   memorised match, not a generalisable policy.
3. **Recurrent students generalise.** Student-1 recurrent in
   particular *improves* with grid size (14.82 -> 16.97 going from
   size 6 to 16); student-2 recurrent stays in the 15.5-17.3
   range across all sizes.
4. **NEAT teacher-1 is more robust than teacher-2.** It degrades
   gracefully (16.58, 16.10, 16.36, 16.40, 12.97) rather than
   collapsing. The "weaker" gen-1 policy is the better generalist.
5. **The best single learned generalist is student-2 recurrent**
   (worst case 15.55, mean 16.40 across sizes). Student-1
   recurrent is a close second (14.82 / 15.99). Both feedforward
   students and both NEAT teachers are dominated by these two
   recurrent students on worst-case foods.

### Interpretation

The Milestone-4 claim that the evolve -> distill loop is
self-improving needs to be split into two claims:

```text
Within training distribution (8x8):
  Teacher-1 16.10  ->  Teacher-2 17.73  ->  Student-2 FF 17.80  (oracle level)

Across all tested grid sizes (worst case):
  Teacher-1 12.97  ->  Teacher-2  0.82  ->  Student-2 FF  1.68  (worse)
  Student-1 rec 14.82  ->  Student-2 rec 15.55                   (slight gain)
```

The second iteration of the loop did not produce a better
policy in any out-of-distribution sense; it produced an
over-specialised in-distribution one. Distilling that specialised
teacher into a feedforward student carried the over-specialisation
into the student. The recurrent student-2 partially recovers
because the GRU compensates for grid-size-dependent dynamics.

Working interpretation:

```text
Iterating evolve -> distill on a fixed environment narrows the
policy onto that environment. Headline numbers at the training
size flatter the loop. Worst-case foods across sizes is a much
harsher and more honest benchmark.

The recurrent student variants are the only learned policies that
remain competitive everywhere. Among them, student-2 recurrent
is best by worst-case foods and by mean foods across sizes.
```

This reframes the "headline" of this repo: matching the oracle on
the training distribution is achievable, but it is not the
interesting metric. The loop must be measured on sizes it has not
seen, or the apparent gain is an artefact of distribution overlap.

### Wall hits across sizes

For completeness, mean wall hits per episode at each size:

```text
Policy                size=6  size=8  size=10  size=12  size=16
greedy_oracle          0.00    0.00    0.00     0.00     0.00
neat_teacher_1         0.92    1.25    0.66     0.58     8.68
student_1_ff           0.00    8.27   10.39    13.76     8.26
student_1_recurrent    2.20    1.57    1.22     0.49     0.46
neat_teacher_2         0.03    0.02    0.49     0.25     0.47
student_2_ff           0.25    0.00    0.66     3.77     6.25
student_2_recurrent    1.58    0.56    0.26     0.27     0.62
```

Recurrent students dominate on safe navigation across all sizes;
feedforward students hit walls heavily once they leave their
training distribution.

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

- The task is still very simple and has a known optimum (the greedy
  oracle).
- Progress shaping is a strong inductive bias.
- The cross-size benchmark exposed brittleness in the iterated-loop
  champion. The "matches the oracle" claim only holds at the
  training grid size; iterated training narrowed the policy onto
  that distribution.
- Seed sensitivity is significant.
- Clean distillation depends strongly on finding a good teacher first.
- The multi-seed gen-2 comparison used n=3 training seeds; the
  combined-strategy improvement is consistent but only ~1.8 SE above
  seeded-only.
- One configuration (seeded + feedforward prior @0.25 at seed 123)
  hit a seed-dependent assertion inside neat-python and is excluded
  from that cell's mean.
- Cross-size evaluation only varied grid size; obstacles and
  stochastic perturbations are still untested.
- Training itself still happens on a fixed 8x8 grid. The cross-size
  numbers test transfer, not robustness-aware training.

## Next steps

```text
1. Train with environment variation (sample grid size per episode
   during evaluation) and see if recurrent-student-2 stays a strong
   generalist or if a new run can beat it.
2. Add obstacles and stochastic action failures to GridWorld and
   re-run the cross-environment benchmark. The greedy oracle stops
   being optimal in those worlds, so the loop has somewhere to go.
3. Use student-2 recurrent (the actual generalist) as the prior
   for a gen-3 evolution and re-run the cross-size benchmark.
4. Wider multi-seed teacher-2 sweep so iteration-2 numbers are not
   conditional on a single best-of-3 gen-2 run.
5. Track performance per parameter across all teachers and students
   at every grid size.
6. Investigate the seed=123 + feedforward-prior + seeded-reseed
   neat-python assertion (innovation tracker collision).
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
