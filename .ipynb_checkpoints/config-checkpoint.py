# config.py

#playertype :random, stable, sta_rand, negative, middle, negative_middle
#agent_type :cfr, dqn
config = {
    "environment": {
        "name": "CustomEnv",
        "max_episode_length": 200,
        "render": False,
        'player_types': ["dqn", "random", "random", "random"],
        'random_probability': 0.7,
        'n_players': 4,
        'n_cards': 15,
    },
    'agent': {
        'learning_rate': 0.003,
        'gamma': 1,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 100,
        'state_size': 80,
        'action_size': 15,
        "batch_size": 64,
        "memory_size": 10000,
    },
    'training': {
        'n_episodes': 100000,
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
