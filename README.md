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

## Initial PoC

The first experiment uses a minimal Gridworld environment.

We compare:

```text
A: NEAT baseline
B: NEAT + energy penalty
C: NEAT + energy-guided local search
D: NEAT + energy-guided local search + distillation
```

Main metric:

```text
performance per parameter
```

Secondary metrics:

- robustness on unseen maps
- energy consumption
- network size
- recovery after perturbation
- behavioral diversity
- stability of internal dynamics

## Architecture

```text
Environment
    ↓
Population of genomes
    ↓
Energy evaluation / relaxation
    ↓
Fitness evaluation
    ↓
Selection + mutation
    ↓
Distillation of elites
    ↓
New population seed
```

## Project philosophy

StrangeLoopGenome treats intelligence not as a static model, but as a recursive developmental process.

The goal is not to build a large model.

The goal is to test whether compact intelligence-like dynamics can be cultivated through evolutionary pressure, energy landscapes, and repeated compression.
