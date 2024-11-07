# config.py

#playertype :random, stable, sta_rand, negavoid, middle, middle_negative

config = {
    "environment": {
        "name": "CustomEnv",
        "max_episode_length": 200,
        "render": False,
        'player_types': ["random", "stable", "sta_rand"],
        'random_probability': 0.7,
    },
    'agent': {
        'learning_rate': 0.001,
        'gamma': 0.50,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 10,
        'state_size': 80,
        'action_size': 15,
    },
    'training': {
        'n_episodes': 200,
        "batch_size": 64,
        "buffer_capacity": 10000,
        "evaluation_frequency": 50,
        'num_experiments': 100,
    },
    "saving": {
        "model_save_path": "saved_models/dqn_agent_model.pth",
        "log_dir": "logs",
        "save_frequency": 100,
    },
    "seed": 42
}
