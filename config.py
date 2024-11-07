# config.py

config = {
    "environment": {
        "name": "CustomEnv",
        "max_episode_length": 200,
        "render": False,
    },
    'agent': {
        "learning_rate": 0.001,
        'gamma': 0.50,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 10,
    },
    'training': {
        'n_episodes': 1000,
        "batch_size": 64,
        "buffer_capacity": 10000,
        "evaluation_frequency": 50,
    },
    "saving": {
        "model_save_path": "saved_models/dqn_agent_model.pth",
        "log_dir": "logs",
        "save_frequency": 100,
    },
    "seed": 42
}
