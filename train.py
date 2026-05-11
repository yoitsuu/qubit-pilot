"""
train.py

Training loop for the qubit-pilot DQN agent.

Ties together the quantum environment and DQN agent:
- Runs episodes where the agent applies gates to reach target state
- Stores experiences in the replay buffer after each step
- Trains the agent after each step once buffer is large enough
- Tracks metrics (rewards, loss, epsilon, fidelity) across episodes
- Saves model checkpoints periodically

Based on the training methodolgy in:
Niu et al. (2019) "Universal quantum control through deep reinforcement learning"
"""

import numpy as np
import torch
from collections import deque

from environment import QuantumEnvironment, GATE_SET
from agent import DQNAgent

#Hyperparameters
from config import CONFIG

#Training Loop
def train():
    print("=" * 60)
    print("qubit-pilot: DQN Quantum Gate Optimization")
    print("=" * 60)
    print(f"Target state: {CONFIG['target_name']}")
    print(f"Noise prob: {CONFIG['noise_prob']}")
    print(f"Max steps: {CONFIG['max_steps']}")
    print(f"Episodes: {CONFIG['n_episodes']}")
    print()

    #Setup
    env = QuantumEnvironment(
        target_name=CONFIG['target_name'],
        max_steps=CONFIG["max_steps"],
        noise_prob=CONFIG['noise_prob'],
    )

    agent = DQNAgent(
        obs_size=env.observation_size,
        n_actions=env.n_actions,
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        epsilon_start=CONFIG['epsilon_start'],
        epsilon_end=CONFIG['epsilon_end'],
        epsilon_decay=CONFIG['epsilon_decay'],
        batch_size=CONFIG["batch_size"],
        target_update_freq=CONFIG['target_update_freq'],
        buffer_capacity=CONFIG['buffer_capacity'],
    )

    #create checkpoint direction
    import os
    os.makedirs("checkpoints", exist_ok=True)

    #Metrics tracking
    episode_rewards = []
    episodes_losses = []
    recent_rewards = deque(maxlen=CONFIG['solved_window'])
    best_avg_reward = float('-inf')

    #Episode loop
    for episode in range(1, CONFIG["n_episodes"] + 1):
        state = env.reset()
        done = False

        ep_reward = 0.0
        ep_losses = []
        ep_steps = 0
        ep_gates = [] #track chosen gates this episode

        #Step loop
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.store(state, action, reward, next_state, done)

            #Learn every N steps
            if ep_steps % CONFIG["learn_every"] == 0:
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)
            
            ep_reward += reward
            ep_gates.append(GATE_SET[action])
            ep_steps += 1
            state = next_state

        #End of every episode
        agent.decay_epsilon()
        
        episode_rewards.append(ep_reward)
        episodes_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        recent_rewards.append(ep_reward)
        avg_reward = np.mean(recent_rewards)

        #Track best avg reward
        best_avg_reward = max(best_avg_reward, avg_reward)

        #Periodic logging
        if episode % CONFIG['print_every'] == 0:
            avg_loss = np.mean(episodes_losses[-CONFIG["print_every"]:]) #takes the last "print_every" episodes then averages them
            print(
                f"Episode {episode:>5} | "
                f"Avg Reward (last {CONFIG['solved_window']}: {avg_reward:.3f}) | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer)}"
            )

        #Periodic checkpointing
        if episode % CONFIG["save_every"] == 0:
            path = CONFIG['checkpoint_path'].format(episode=episode)
            agent.save(path)
        
        #Solved check
        #Consider the task solved if the agent consistently gets
        #high reward over the last "solved_window" episodes
        #for example, solved_reward of 1.8 means fidelity of
        # ~0.8 + solve bonus ~1.0 consistently.
        # We interpret this as the agent has learned a reliable policy.
        if avg_reward >= CONFIG["solved_reward"] and episode >= CONFIG["solved_window"]:
            print(f"\nSolved at episode {episode}! Avg reward: {avg_reward:.3f}")
            agent.save('checkpoints/dqn_solved.pt')
            break
    
    print(f"\nTraining complete. Best avg reward: {best_avg_reward:.3f}")
    agent.save("checkpoints/dqn_final.pt")

    #Save training metrics for evaluate.py
    import json
    with open("checkpoints/metrics.json", "w") as f:
        json.dump({
            "rewards": episode_rewards,
            "losses": episodes_losses,
        }, f)

    return episode_rewards, episodes_losses, agent


#Entry point
if __name__ == "__main__":
    rewards, losses, agent = train()

    #Quick summary of what trained agent does
    print("\n--- Trained Agent Demo ---")
    env = QuantumEnvironment(
        target_name=CONFIG["target_name"],
        noise_prob=CONFIG['noise_prob'],
    )

    #Run 5 demo episodes with greedy policy (no exploration)
    agent.epsilon = 0.0
    for i in range(5):
        state = env.reset()
        done = False
        gates = []
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            gates.append(GATE_SET[action])
            total_reward += reward

        env.render() #render so we can see

        print(f"Demo {i + 1}: gates={gates}, reward = {total_reward:.3f}")

        #auto eval for plots
        from evaluate import evaluate
        evaluate()