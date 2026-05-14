# config.py
# single source of truth for all hyperparameters

CONFIG = {
    # Environment
    "target_name":        "|i>",

    #source for h/x/s: https://arxiv.org/pdf/2302.08690 (clifford gates)
    #source for z: https://www.nature.com/articles/s41598-022-10339-0 (seems to be a "virtual" phase shift, so near-zero noise)
    #source for t: it seems that direct benchmarking is non-trivial, so
    #per this source we can estimate at ~5x clifford rate https://arxiv.org/pdf/2008.09503
    #source for id: https://arxiv.org/pdf/1406.3364 (treating ID gate noise as accumulated decoherence during one gate duration)
    "gate_noise": {
        "x": 0.001,
        "h": 0.001,
        "t": 0.005,
        "s": 0.001,
        "z": 0.0,
        "id": 0.0001
    },
    "max_steps":          10,

    # Training
    "n_episodes":         2000,
    "learn_every":        1,

    # Agent
    "lr":                 1e-3,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.998,
    "batch_size":         64,
    "target_update_freq": 50,
    "buffer_capacity":    10_000,

    # Logging and saving
    "print_every":        100,
    "save_every":         500,
    "checkpoint_path":    "checkpoints/dqn_{episode}.pt",
    "final_checkpoint": "checkpoints/dqn_final.pt",
    "rolling_window":     100,

    # Evaluation
    "n_eval_episodes":    200,
    "output_dir":         "plots",
}