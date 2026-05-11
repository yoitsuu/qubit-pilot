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
from qiskit.quantum_info import Statevector, state_fidelity

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

        self.target_name = target_name
        self.target_state = TARGET_STATES[target_name]
        self.max_steps = max_steps
        self.noise_prob = noise_prob
        self.fidelity_threshold = fidelity_threshold
        
        #Noise sim
        self.simulator = AerSimulator(method="statevector")
        self.noise_model = self._build_noise_model(noise_prob)

        #Episode vars
        self.current_circuit = None
        self.current_step = 0

        #RL Interface
        def reset(self) -> np.ndarray:
            """
            Reset to a new episode. Qubit set to |0>.
            Return the initial state observation.
            """
            self.current_circuit = QuantumCircuit(1)
            self.current_step = 0
            return self.state_vector_to_obs(np.array([1, 0], dtype=complex))
        
        def step(self, action: int) -> tuple[np.ndarray, float, bool]:
            """
            Apply one gate (chosen by the agent) to the qubit.

            Parameters
            action : int
                Index into GATE_SET (0 = 'x', 1 = 'h', etc.)
            
            Returns
            obs: np.ndarray, shape(4,)
                New state observation [re(a0), im(a0), re(a1), im(a1)]
            reward: float
                Fidelity between new state and target state (0.0 to 1.0)
            done: bool
                Whether the episode ended
            """

            #Apply the chosen gate to circuit
            gate_name = GATE_SET[action]
            self._apply_gate(gate_name)
            self.current_step += 1

            #sim and update state vector
            statevector = self._get_statevector()

            #check fidelity
            fidelity = float(state_fidelity(
                Statevector(statevector),
                Statevector(self.target_state)
            ))

            #Reward
            #Big reward for solving
            #raw fidelity per step otherwise, to encourage progress
            solved = fidelity >= fidelity_threshold
            reward = fidelity + (1.0 if solved else 0.0)

            done = solved or (self.current_step >= self.max_steps)

            obs = self._statevector_to_obs(statevector)
            return obs, reward, done
        
        #Helper functions
        def _apply_gate(self, gate_name: str):
            "Append named gate to current circuit"
            gate_fn = getattr(self.current_circuit, gate_name)
            gate_fn(0) #apply to qubit 0
        
        def _get_statevector(self) -> np.ndarray:
            """
            Simulate current circuit and return statevector

            Notes:
            use save_statevector() instead of measure_all() because we want the full complex vector,
            not a collapsed observation
            """
            qc = self.current_circuit.copy()
            qc.save_statevector()

            result = self.simulator.run(
                qc,
                noise_model = self.noise_model,
            ).result()

            sv = result.get_statevector(qc)
            return np.array(sv, dtype=complex)
        
        def _statevector_to_obs(self, sv: np.ndarray) -> np.ndarray:
            """
            Convert complex statevector to real-valued observation vector.

            Notes:
            A single qubit statevector has 2 complex altitudes: [a0, a1]
            Split into real and imaginary parts -> 4 floats in total.
            The neural network will receive this as input.

            Example: [0.707 + 0j, 0.707 + 0j] = [0.707, 0.0, 0.707, 0.0]
            """
            return np.array([
                sv[0].real, sv[0].imag,
                sv[1].real, sv[1].imag,
            ], dtype=np.float32)
        
        def _build_noise_model(self, prob: float) -> NoiseModel:
            """
            Build the depolarizing noise model.

            Notes:
            Depolarozing noise is the simplest realistic noise model. After each
            gate, with probability "prob", the qubit is replaced by a completely random state.
            Supposedly this is how real quantum hardware behaves.

            At prob = 0.01 (1%), the circuit is very noisy. A 10-gate sequence has ~10% chance of at least 1 error.
            I expect the agent will be encouraged to find shorter, more efficient gate sequences rather than brute force.
            """
            noise_model = NoiseModel()
            error = depolarizing_error(prob, 1) #1-qubit depolarizing error
            noise_model.add_all_qubit_quantum_error(error, GATE_SET)
            return noise_model

        #Utility Section
        
        @property
        def observation_size(self) -> int:
            #Size of state vector the agent receives (input to Neural Net)
            return 4 #[re(a0), im(a0), re(a1), im(a1)]
        
        @property
        def n_actions(self) -> int:
            #number of discrete actions available
            return N_ACTIONS
        
        def render(self):
            #Print current circuit and statevector for debugging
            print(f"\nStep {self.current_step} | Target: {self.target_name}")
            print(self.current_circuit.draw(output="text"))
            sv = self._get_statevector
            fidelity = float(state_fidelity(
                Statevector(sv), Statevector(self.target_state)
            ))
            print(f"Statevector: {np.round(sv, 3)}")
            print(f"Fidelity: {fidelity:.4f}")
        
        if __name__ == "__main__":
            env = QuantumEnvironment(target_name="|+>", noise_prob=0.01)
            obs = env.reset()
            print("Initial obs:", obs)
            obs, reward, done = env.step(1)  # apply H gate
            env.render()
            print(f"Reward: {reward:.4f}, Done: {done}")