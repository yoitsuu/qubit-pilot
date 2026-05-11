# config.py
# single source of truth for all hyperparameters

CONFIG = {
    # Environment
    "target_name":        "|+>",
    "noise_prob":         0.01,
    "max_steps":          10,

    # Training
    "n_episodes":         2000,
    "learn_every":        1,

    # Agent
    "lr":                 1e-3,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.995,
    "batch_size":         64,
    "target_update_freq": 50,
    "buffer_capacity":    10_000,

    # Logging and saving
    "print_every":        100,
    "save_every":         500,
    "checkpoint_path":    "checkpoints/dqn_{episode}.pt",
    "solved_reward":      0.5,
    "solved_window":      200,
    "rolling_window":     100,

    # Evaluation
    "n_eval_episodes":    200,
    "output_dir":         "plots",
}