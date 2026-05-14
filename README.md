# qubit-pilot

A deep reinforcement learning agent that learns to prepare target quantum states by discovering optimal gate sequences, trained on a noisy qubit simulator built with Qiskit and PyTorch.

Inspired by [Niu et al. (2019) "Universal quantum control through deep reinforcement learning"](https://arxiv.org/pdf/1803.01857).

---

## Overview

Quantum state preparation is a fundamental challenge in near-term quantum computing. Given a qubit initialized to |0⟩, the task is to find a sequence of gates that transforms it into a target state with high fidelity under realistic hardware noise.

Rather than using gradient-based optimization, this project trains a Deep Q-Network (DQN) to discover gate sequences autonomously through trial and error. The agent learns entirely from a sparse reward signal: it receives no feedback until it successfully prepares the target state, forcing it to explore the gate space efficiently.

The environment applies depolarizing noise after each gate, modeling the imperfect hardware of NISQ-era devices. This pressure encourages the agent to find short, efficient sequences rather than brute-forcing long ones, since every additional gate compounds error.

---

## Results

Training on the |i⟩ target state (`[1/√2, i/√2]`), a Y-basis state requiring
both amplitude and phase manipulation, the agent discovers the optimal 2-gate
solution: Hadamard followed by S gate.
```
Episode  500 | Avg Reward: 1.380 | Epsilon: 0.368
Episode 1000 | Avg Reward: 1.692 | Epsilon: 0.135
Episode 1500 | Avg Reward: 1.662 | Epsilon: 0.050
Episode 2000 | Avg Reward: 1.692 | Epsilon: 0.050

Mean fidelity (200 eval episodes): 1.000
Fidelity >= 0.99: 100.0% of episodes
Mean episode length: 2.00 gates

```

The agent learned to apply H at step 1 and S at step 2. This is the provably optimal 
sequence. The agent was not told the answer or the gate ordering. Notably, the 
agent had to learn that ordering matters: S → H produces a different state 
entirely.

Gate noise is modeled per-gate based on superconducting transmon hardware characteristics. See
config.py for citations and noise settings.

To train on a different target, change `target_name` in `config.py` and rerun. 
If introducing a new target state (`TARGET_STATES` in `environment.py`), just add it as a key-value pair in that object.

---

## How It Works

### Environment (`environment.py`)

The quantum environment follows a standard RL interface: `reset()` and `step(action) -> (state, reward, done)`.

- **State:** The current qubit statevector, represented as 4 floats — real and imaginary parts of the two complex amplitudes
- **Actions:** 6 discrete gates — X, H, T, S, Z, Identity
- **Reward:** Sparse — a solve bonus plus remaining-step savings when fidelity exceeds 0.99, a small per-step penalty otherwise
- **Noise:** Depolarizing error applied after every gate via Qiskit Aer's noise model

The statevector is extracted directly from the simulator (no measurement collapse) so the agent observes the full quantum state at each step.

### Agent (`agent.py`)

A standard DQN implementation in PyTorch with two key stabilization techniques:

- **Replay buffer:** Stores past experiences and samples random batches for training, breaking temporal correlations that would otherwise destabilize learning
- **Target network:** A periodically-synced frozen copy of the online network, used to compute stable Q-value targets and prevent the moving-target problem

Action selection uses epsilon-greedy exploration starting fully random and decaying toward greedy exploitation as the Q-function becomes accurate.

### Training (`train.py`)

The training loop runs 2000 episodes. The agent learns after every step once the replay buffer has accumulated enough experiences. Metrics are logged every 100 episodes and checkpoints are saved every 500.

After training completes, evaluation and visualization run automatically.

---

## Project Structure

```
qubit-pilot/
├── config.py         # single source of truth for all hyperparameters
├── environment.py    # Qiskit quantum environment
├── agent.py          # DQN agent with replay buffer and target network
├── train.py          # training loop, demo, and evaluation pipeline
├── evaluate.py       # visualization — training curve, gate heatmap, fidelity
├── checkpoints/      # saved model weights and training metrics
└── plots/            # evaluation figures
```

---

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yoitsuu/qubit-pilot.git
cd qubit-pilot
uv sync
```

For GPU support (CUDA 12.1):
```bash
uv add torch --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

---

## Usage

Train the agent and run evaluation:
```bash
uv run python train.py
```

Re-run evaluation on a saved checkpoint without retraining:
```bash
uv run python evaluate.py
```

---

## Configuration

All hyperparameters live in `config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `target_name` | `\|i>` | Target quantum state. Options: `\|0>`, `\|1>`, `\|+>`, `\|->`, `\|i>` |
| `noise_prob` | `0.01` | Depolarizing error probability per gate |
| `max_steps` | `10` | Maximum gates per episode |
| `n_episodes` | `2000` | Training episodes |
| `epsilon_decay` | `0.998` | Exploration rate decay per episode |

To train on a different target, change `target_name` in `config.py` and run `train.py`.
If the new target is not present in `TARGET_STATES` in `environment.py`, simply add the key-value pair.

---

## Connection to Research

This project is a simplified implementation of the core RL-for-quantum-control framework explored in Niu et al. (2019), which demonstrated that deep RL agents can discover quantum control policies that outperform gradient-based baselines by orders of magnitude in gate fidelity.

The key design decisions here mirror that work: sparse reward to encourage efficient sequences, depolarizing noise to model NISQ hardware, and a discrete gate set that spans the single-qubit Clifford group (and a few extras). The extension to multi-qubit systems, continuous pulse control, and more complex noise models represents natural next steps toward the problem framing in Niu's current research on quantum pulse processing.

---

## Dependencies

- [Qiskit](https://qiskit.org/) — quantum circuit construction and simulation
- [Qiskit Aer](https://qiskit.github.io/qiskit-aer/) — noisy statevector simulation
- [PyTorch](https://pytorch.org/) — DQN implementation
- [NumPy](https://numpy.org/) — numerical operations
- [Matplotlib](https://matplotlib.org/) — evaluation visualization