from environment.multiplayer_hagetaka_env import MultiPlayerHagetakaEnv
from agent.cfr_agent import CFRPlayer
from train.train_agent import train_agent
from config import config

env = MultiPlayerHagetakaEnv(config['environment']['n_players'], config['environment']['n_cards'], config['environment']['player_types'])

players = env.players


train_agent(env, players, config['training']['n_episodes'], config['training']['num_experiments'])