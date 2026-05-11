"""
agent.py

The agent observes the current qubit statevector (4 floats) and selects which gate
to apply next. It learns by storinf experiences in a replay
buffer and training a Q-network to predict cumulative rewards.

Architecture notes:
- Small feedforward network (3 layers): state space is only 4D,
so I don't think we need anything deep. I suspect a bigger network
is likely to overfit

- Replay buffer: breaks temporal correlations between consecutive
experiences. The goal is to stabilize training with this.

- Epsilon-greedy exploration: start random, and gradually exploit
the learned Q-function as it becomes more accurate.

- Two networks (online + target): the target network is a frozen copy
of the online network, updated periodically. This prevents the Q-value
targets from shifting every step, which would cause instability (like
hitting a moving bullseye). Largely based on Deepmind's original DQN paper.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#Q-Network
class QNetwork(nn.Module):
    """
    Feedforward neural network that approximates the Q-function.

    Input: state observation (4 flaots - real/imag parts of statevector)
    Output: Q-value for each possible action (one per gate in GATE_SET)

    The Q-value for action A in state S estimates:
    "How much total future reward will I get if I take action A here?"

    Architecture: 3 fully connected layers wiht ReLU activations.
    No softmax on output, since I want raw Q-values instead of probabilities.
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 64):
        super().__init()

        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

#Replay Buffer
class ReplayBuffer:
    """
    Circular buffer storing past (state, action, reward, next_state, done)
    experience tuples.

    Why I have this:
    If we trained on each experience immediately after collecting it,
    consecutive experiences would be highly correlated. For example, after
    the agent applies gate H, the next state is always an H-rotated state.
    Training on correlated data tends to cause overfit on recent experience, and
    forget earlier lessons (catastraphic forgetting).

    By storing experiences and sampling random batches, I will break that correlaton
    to ideally get more stable, generalizable training.

    
    Parameters
    capacity : int
        Maximum number of experiences to store. Once full, oldest experiences
        are overwritten (deque struct does this easily).
    """

    def __init__(self, capacity: int = 10_000): #ten thousand, underscore is only visual
        self.buffer = deque(maxlen=capacity)
    
    def store(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        #save 1 experience tuple
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        #Sample a random batch of experiences
        #Return separate arrays for each component, makes the torch work easier
        batch = random.sample(self.buffer, batch_size)

        #unzip the list of tuples into separate arrays
        # * "unpacks" batch into separate tuples into 1 giant list of tuples
        # zip combines the elements by position
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    

#DQN Agent
class DQNAgent:
    """
    Deep Q-Network agent for quantum gate sequence optimization.

    The agent maintains 2 networks:
    - online_net: updated every learning step via backprop
    - target_net: a periodically synced frozen copy of online_net,
    used to compute stable Q-value targets

    Without the target network, the system would compute targets using the same
    network it updates. As per the original DQN paper by Deepmind, I chose to
    use 2 networks to resolve this issue.

    Parameters
    obs_size : int
        Size of the state observation vector (4 for single qubit)
    n_actions : int
        Number of discrete actions (originally 6 gates in GATE_SET)
    lr: float
        Learning rate for the Adam optimizer
    gamma : float
        Discount factor. How much to value future rewards vs immediate ones.
        0.0 = only care for immediate reward
        1.0 = value all future rewards equally
        0.99 is standard slightly discount future
    epsilon_start: float
        Initial exploration rate. 1.0 = always explore randomly at first
    epsilon_end : float
        Minimum exploration rate. Set a lower bound so the agent does not
        go fully greedy (avoid getting stuck)
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each episode.
        0.995 means epsilon halves roughly every 140 episodes.
    batch_size : int
        Number of experiences to sample per learning step.
    target_update_freq : int
        How many learning steps between target network syncs.
    buffer_capacity : int
        Maximum replay buffer size
    """

    def __init__(
            self,
            obs_size: int,
            n_actions: int,
            lr: float = 1e-3,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float =0.995,
            batch_size: int = 64,
            target_update_freq: int = 50,
            buffer_capacity: int = 10_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        #Use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        #Two networks
        self.online_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict()) #init same "random" weights
        self.target_net.eval() #never trains, only copy weights, so can set to eval mode at init

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        #Count learning steps for target network sync
        self.learn_steps = 0

    #Action selection
    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon: pick a random gate (explore).
        Otherwise: pick the gate with highest Q-value (exploit).

        Early in training epsilon is high, so we explore mostly
        randomly. As epsilon decays, we trust Q-function more.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        #convert state to tensor, run through online network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #fake a batch size of 1 for tensor shape
        with torch.no_grad():
            q_values = self.online_net(state_tensor) #no need to save backprop for the single feed op here
        
        #pick action with highest Q-value
        return int(q_values.argmax(dim=1).item())
    
    def decay_epsilon(self):
        #decay epsilon after each episode. called by training loop
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    #Experience storage
    def store(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        #Store one experience in the replay buffer
        self.buffer.store(state, action, reward, next_state, done)
    
    #Learning
    def learn(self) -> float | None:
        """
        Sample a batch from the replay buffer and perform one gradient update.

        Return the loss value for logging, or None if the buffer is too small
        to sample a full batch.

        Notes: The Bellman equation drives learning here.
        Q(s, a) = r + gamma * max_a'[ Q_target(s', a') ] * (1 - done)

        In english: the true Q-value of taking action a in state s is the immediate reward r,
        plus the discounted best Q-value we can get from the next state s'. The (1 - done) term zeroes
        out future rewards when the episode is over (there is no next state after done = True).

        The model minimizes the MSE between the online network's Q-value predictions
        and the Bellman targets.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        #Move everything to the gpu/cpu
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        #Current Q-values from online network
        # get Q-values for all actions -> select the one we actually take
        q_values = self.online_net(states) #output (64, 6) 1 q_vals per gate per state, assuming 6 gates in set
        #we're indexing into q-values which is dim=2, hence actions needs to be dim=2 as well
        #make actions (64,) -> (64, 1) with unsqueeze so we can use it as an index
        #hardcode 1 as the dimension, since q-vals is (batch, actions) and we want to go through actions
        #squeeze 1 at the end to change back to (64,) so we just have the q-values, no extra dim
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        """
        Concrete example of how the gather snippet works
        q_values = tensor([
            [0.2, 0.8, 0.1, 0.3], experience 0
            [0.5, 0.1, 0.9, 0.2], experience 1
            [0.3, 0.4, 0.2, 0.7], experience 2
        ]) shape (3, 4)

        actions = tensor([
            [1], experience 0 took action 1
            [2], experience 1 took action 2
            [3], experience 2 took action 3
        ]) shape (3, 1)

        q_values.gather(1, actions)
        ->
        tensor([
            [0.8], picked index 1 from row 0
            [0.9], picked index 2 from row 1
            [0.7], picked index 3 from row 2
        ]) shape (3, 1)
        """

        #target q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values #max dim=1 so we select only the best q-value
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        #MSE loss between predicted/target q vals
        loss = nn.MSELoss()(q_values, targets)

        #Backprop
        self.optimizer.zero_grad()
        loss.backward()
        #Gradient clipping to prevent exploding gradients, standard DQN practice afaik
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        #periodically sync target network with online network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return float(loss.item())
    
    #Utility
    def save(self, path: str):
        #save the online network weights to disk
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        #load network weights from disk
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"Model loaded from {path}")
    
#Quick sanity check, from running file directly
if __name__ == "__main__":
    from environment import QuantumEnvironment
 
    env = QuantumEnvironment(target_name="|+>", noise_prob=0.01)
    agent = DQNAgent(obs_size=env.observation_size, n_actions=env.n_actions)
 
    # Run one random episode to verify everything connects
    state = env.reset()
    done = False
    total_reward = 0.0
 
    print("\nRunning one random episode...")
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        print(f"  Action: {action} ({['x','h','t','s','z','id'][action]}), "
              f"Reward: {reward:.4f}, Done: {done}")
 
    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Buffer size: {len(agent.buffer)}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"\nSanity check passed — agent and environment are connected correctly.")
