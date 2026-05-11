"""
environment.py

The environment wraps a single-qubit quantum system simulated with Qiskit.
The RL interface is intentionally simple: reset(), step(action) -> (state, reward, done)

The goal of the RL agent is to learn a sequence of gates that transforms the initial
qubit state |0> into a target state with high fidelity. In order to make it more realistic,
I added depolarizing noise after each gate.

This implementation is largely based on: 
Niu et al. (2019) "Universal quantum control through deep reinforcement learning"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantim_info import Statevector, state_fidelity

"""
Gate Definitions
The following gates are actions available to our agent as single-qubit gates.
The state is intentionall small to maintain a manageably sized action space for DQN.

X - flips |0> to |1> and vice versa
H - creates superposition state
T - applies a phase rotation of pi/4
S - applies a phase rotation of pi/2
Z- flips the phase of |1> (doesn't do anything to |0>)
I - identity (lets agent try out "nothing" actions)

Notes for self:
X, H, T form a basis to reach any single-qubit state.
S and Z allow for more optimal paths in circuit design.
I lets the agent try out doing nothing
"""
GATE_SET = ["x", "h", "t", "s", "z", "id"]
N_ACTIONS = len(GATE_SET)

"""
Target States
The states we want the agent to learn how to reach.
I decided to use numpy complex state vectors of length 2 for the representation of these.

|0>: np.array([1, 0], dtype=complex)
|1>: np.array([0, 1], dtype=complex)
|+>: np.array([1, 1], dtype=complex) / np.sqrt(2)
|->: np.array([1, -1], dtype=complex) / np.sqrt(2)
|i>: np.array([1, 1j]), dtype=complex) / np.sqrt(2)
"""
TARGET_STATES = {
    "|0>": np.array([1, 0], dtype=complex),
    "|1>": np.array([0, 1], dtype=complex),
    "|+>": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|->": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|i>": np.array([1, 1j], dtype=complex) / np.sqrt(2),
}

class QuantumEnvironment:
    """
    A single-qubit quantum environment for reinforcement learning.

    The agent starts with qubit in state |0> and applies 1 gate at a time.
    After each gate, the environment returns:
    - state: the current state vector (real + imaginary of both amplitudes)
    - reward: fidelity between current state and target state
    - done: boolean if max_steps reached or fidelity >= fidelity_threshold

    Parameters:
    target_name: str
        Target state we are aiming for, from the keys in TARGET_STATES
    max_steps: int
        Maxiumum number of gates the agent can apply per episode
    noise_prob: float
        Depolarizing error probability per gate (0.0 = noiseless, 0.01 = 1%, etc)
    fidelity_threshold: float
        Fidelity at which we consider the task solved (can end episodes early)
    """

    def __init__(
            self,
            target_name: str = "|+>",
            max_steps: int = 10,
            noise_prob: float = 0.01,
            fidelity_threshold: float = 0.99,
    ):
        assert target_name in TARGET_STATES, (
            f"Unknown target '{target_name}'. Choose from {list(TARGET_STATES.keys())}"
        )