"""
evaluate.py

Evaluation and visualization for the trained qubit-pilot DQN agent.

Loads a saved checkpoint and produces four plots:
1. Training curve — reward per episode with rolling average
2. Epsilon decay — explore/exploit tradeoff over training
3. Gate frequency heatmap — which gates the agent chose at each step
4. Fidelity distribution — reliability of the learned policy

Run after training:
    uv run python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from environment import QuantumEnvironment, GATE_SET
from agent import DQNAgent

# Config
from config import CONFIG


# ---------------------------------------------------------------------------
# Helper: run evaluation episodes with greedy policy
# ---------------------------------------------------------------------------

def run_eval_episodes(agent: DQNAgent, env: QuantumEnvironment, n: int):
    """
    Run n episodes with epsilon=0 (fully greedy, no exploration).
    Returns per-episode data for plotting.
    """
    agent.epsilon = 0.0

    all_gates     = []   # list of gate sequences per episode
    all_fidelities = []  # final fidelity per episode
    all_rewards   = []   # total reward per episode
    all_lengths   = []   # number of steps per episode

    for _ in range(n):
        state = env.reset()
        done  = False
        ep_gates    = []
        ep_reward   = 0.0
        final_fidelity = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            ep_gates.append(GATE_SET[action])
            ep_reward += reward
            state = next_state

        # Get final fidelity via render data
        from qiskit.quantum_info import Statevector, state_fidelity
        sv = env._get_statevector()
        final_fidelity = float(state_fidelity(
            sv, Statevector(env.target_state)
        ))

        all_gates.append(ep_gates)
        all_fidelities.append(final_fidelity)
        all_rewards.append(ep_reward)
        all_lengths.append(len(ep_gates))

    return all_gates, all_fidelities, all_rewards, all_lengths


# ---------------------------------------------------------------------------
# Helper: reconstruct training metrics by re-running training curve
# We approximate epsilon decay since we don't store it during training
# ---------------------------------------------------------------------------

def reconstruct_training_metrics():
    """
    Reconstruct approximate training metrics for plotting.
    We re-run a short training pass to get the reward/loss curves.
    """
    import json
    with open("checkpoints/metrics.json") as f:
        metrics = json.load(f)
    episode_rewards = metrics["rewards"]
    episode_losses = metrics["losses"]
    return episode_rewards, episode_losses


# ---------------------------------------------------------------------------
# Plot 1: Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(ax, rewards: list, window: int = 100):
    """
    Plot raw episode rewards and a smoothed rolling average.
    The rolling average reveals the learning trend beneath noisy episodes.
    """
    episodes = np.arange(1, len(rewards) + 1)
    raw = np.array(rewards)

    # Rolling average
    rolling = np.convolve(raw, np.ones(window) / window, mode="valid")
    rolling_episodes = episodes[window - 1:]

    ax.plot(episodes, raw, alpha=0.3, color="#4C9BE8", linewidth=0.8, label="Episode reward")
    ax.plot(rolling_episodes, rolling, color="#1A5FA8", linewidth=2.0,
            label=f"Rolling avg (n={window})")

    ax.axhline(y=1.9, color="#E85C4C", linestyle="--", linewidth=1.2, label="Optimal reward (1.9)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 2: Epsilon decay
# ---------------------------------------------------------------------------

def plot_epsilon_decay(ax, n_episodes: int, epsilon_start: float = 1.0,
                       epsilon_end: float = 0.05, epsilon_decay: float = 0.995):
    """
    Reconstruct and plot the epsilon decay curve.
    Shows how exploration rate decreased over training.
    """
    epsilons = []
    eps = epsilon_start
    for _ in range(n_episodes):
        epsilons.append(eps)
        eps = max(epsilon_end, eps * epsilon_decay)

    ax.plot(np.arange(1, n_episodes + 1), epsilons, color="#4CAF82", linewidth=2.0)
    ax.axhline(y=epsilon_end, color="#E85C4C", linestyle="--", linewidth=1.2,
               label=f"Floor ({epsilon_end})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon) Decay")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)


# ---------------------------------------------------------------------------
# Plot 3: Gate frequency heatmap
# ---------------------------------------------------------------------------

def plot_gate_heatmap(ax, all_gates: list, max_steps: int):
    """
    Heatmap of gate choices across step positions.

    Rows = gates, Columns = step positions.
    Cell value = fraction of episodes where that gate was chosen at that step.

    This reveals the structure of the learned policy:
    - If the agent learned correctly, H should dominate step 0
    - Later steps should show identity or nothing (episode already solved)
    """
    n_gates  = len(GATE_SET)
    n_steps  = max_steps
    counts   = np.zeros((n_gates, n_steps))

    for ep_gates in all_gates:
        for step_idx, gate in enumerate(ep_gates):
            gate_idx = GATE_SET.index(gate)
            counts[gate_idx, step_idx] += 1

    # Normalize to fractions
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # avoid division by zero
    fractions = counts / col_sums

    im = ax.imshow(fractions, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of episodes")

    ax.set_xticks(np.arange(n_steps))
    ax.set_xticklabels([f"Step {i+1}" for i in range(n_steps)], fontsize=7)
    ax.set_yticks(np.arange(n_gates))
    ax.set_yticklabels([g.upper() for g in GATE_SET])
    ax.set_title("Gate Frequency by Step Position")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gate")

    # Annotate cells with percentage
    for i in range(n_gates):
        for j in range(n_steps):
            val = fractions[i, j]
            if val > 0.05:  # only annotate meaningful cells
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.6 else "black")


# ---------------------------------------------------------------------------
# Plot 4: Fidelity distribution
# ---------------------------------------------------------------------------

def plot_fidelity_distribution(ax, fidelities: list):
    """
    Histogram of final fidelities across evaluation episodes.
    A well-trained agent should be heavily concentrated near 1.0.
    """
    ax.hist(fidelities, bins=20, range=(0, 1), color="#4C9BE8",
            edgecolor="white", linewidth=0.5)

    mean_fid = np.mean(fidelities)
    ax.axvline(x=mean_fid, color="#E85C4C", linestyle="--", linewidth=1.5,
               label=f"Mean fidelity: {mean_fid:.3f}")

    ax.set_xlabel("Final Fidelity")
    ax.set_ylabel("Number of Episodes")
    ax.set_title(f"Fidelity Distribution ({len(fidelities)} eval episodes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(0, 1.05)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    print("=" * 60)
    print("qubit-pilot: Evaluation & Visualization")
    print("=" * 60)

    # --- Load environment and agent ---
    env = QuantumEnvironment(
        target_name=CONFIG["target_name"],
        noise_prob=CONFIG["noise_prob"],
        max_steps=CONFIG["max_steps"],
    )

    agent = DQNAgent(
        obs_size=env.observation_size,
        n_actions=env.n_actions,
    )

    print(f"Loading checkpoint: {CONFIG['checkpoint_path']}")
    agent.load(CONFIG['final_checkpoint'])

    # --- Run evaluation episodes ---
    print(f"Running {CONFIG['n_eval_episodes']} greedy evaluation episodes...")
    all_gates, fidelities, rewards, lengths = run_eval_episodes(
        agent, env, CONFIG["n_eval_episodes"]
    )

    print(f"Mean fidelity:      {np.mean(fidelities):.4f}")
    print(f"Fidelity >= 0.99:   {np.mean(np.array(fidelities) >= 0.99):.1%} of episodes")
    print(f"Mean episode length:{np.mean(lengths):.2f} gates")
    print(f"Mean reward:        {np.mean(rewards):.4f}")

    # --- Re-run training to get reward/loss history for plotting ---
    # Note: this re-trains from scratch to collect metrics.
    # In a real project you'd save metrics during training.
    print("\nLoading training metrics...")
    episode_rewards, episode_losses, _ = reconstruct_training_metrics()

    # --- Build figure ---
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"qubit-pilot: DQN Quantum Gate Optimization\n"
        f"Target: {CONFIG['target_name']} | Noise: {CONFIG['noise_prob']} | "
        f"Mean Fidelity: {np.mean(fidelities):.3f}",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_training_curve(ax1, episode_rewards, window=CONFIG["rolling_window"])
    plot_epsilon_decay(ax2, n_episodes=len(episode_rewards))
    plot_gate_heatmap(ax3, all_gates, max_steps=CONFIG["max_steps"])
    plot_fidelity_distribution(ax4, fidelities)

    # --- Save and show ---
    output_path = os.path.join(CONFIG["output_dir"], "evaluation.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()