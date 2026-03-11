# FindWay-RL

A PPO-based pathfinding project on random-obstacle GridWorlds, featuring curriculum-style training, checkpoint resume, batch evaluation, and pygame visualization.

The agent starts at the top-left corner of a `10x10` grid and must reach the bottom-right corner. Obstacles are randomly generated, but every map is guaranteed to be reachable through BFS-based filtering.

---

## Features

- Random reachable GridWorld generation
- PPO training with shared actor-critic MLP
- Curriculum-style two-stage training
- Fast vectorized numpy environment
- GPU rollout buffers + AMP support
- Checkpoint save/resume
- Batch evaluation script
- Pygame visualization

---

## Environment

### Task
The agent navigates a square grid:

- **Start:** `(0, 0)`
- **Goal:** `(size - 1, size - 1)`

### Actions
- `0`: up
- `1`: down
- `2`: left
- `3`: right

### Reward
- Valid move: `-0.01`
- Invalid move / hitting wall / obstacle: `-0.05`
- Reaching the goal: `+1.0`

### Observation
The observation vector contains:

- Agent position
- Goal position
- Flattened obstacle map

---

## Training Strategy

This project uses a simple curriculum-style training pipeline:

1. **Pretraining**
   - Train on easier maps (for example, obstacle density `0.20`)
   - Learn basic pathfinding behavior

2. **Finetuning**
   - Continue training on a density range (currently `0.20 ~ 0.40`)
   - Improve robustness and generalization

When a density range is used, each environment resamples its obstacle density after reset, so the policy sees more diverse map distributions during training.

---

## Project Structure

```text
.
├── train.py
├── eval.py
├── ppo_checkpoint.pt
└── README.md
