import matplotlib.pyplot as plt
import numpy as np
import random
import dill
from config import config

from environment.multiplayer_hagetaka_env import MultiPlayerHagetakaEnv
from agent.cfr_agent import CFRPlayer
from agent.dqn_agent import DQNPlayer

from train.evaluation import (
    calculate_mean_scores_and_ranks,
    plot_mean_scores,
    plot_mean_ranks,
    plot_average_first_place_rate,
    print_last_10_episodes,
)

n_cards = config['environment']['n_cards']
player_types = config['environment']['player_types']
# トレーニングループ
def train_agent(env, players, n_episodes, num_experiments):
    
    # 強化学習エージェントのロード
    for i, player in enumerate(players):
        if isinstance(player, CFRPlayer):
            player = CFRPlayer.load("saved_models/cfr_agent_player.dill")
            if player is None:
                print(f"Creating a new agent for player {i}.")
                player = CFRPlayer(
                    player_index=i,
                    n_cards=env.n_cards,
                    n_players=env.n_players
                )
            players[i] = player
        elif isinstance(player, DQNPlayer):
            player = DQNPlayer.load("saved_models/dqn_agent_model.pth")
            if player is None:
                print(f"Creating a new agent for player {i}.")
                player = DQNPlayer(
                    player_index=i,
                    n_cards=env.n_cards,
                    n_players=env.n_players
                )
    player_scores = np.zeros((env.n_players, n_episodes))  # 各プレイヤーのスコアを記録する
    total_rewards = []  # 各エピソードでの累積報酬を保存するリスト
    last_10_episodes = []  # ラスト10エピソードのカード履歴を保存するリスト
    
    scores_history = np.zeros((num_experiments, n_episodes, env.n_players))
    ranks_history = np.zeros((num_experiments, n_episodes, env.n_players), dtype=int)   
    
    for experiment in range(num_experiments):
        print(f"Running experiment {experiment + 1}/{num_experiments}")
        for iteration in range(n_episodes):
            state = env.reset()  # 環境の初期化
            for i, player in enumerate(players):
                players[i].reset()
            done = False  # ゲーム終了フラグ
            total_reward = 0
            
            print((f"State length: {len(state)}, Expected: {env.state_size}"))
            
            score_card_history = []  # 各エピソードの得点カードの順番
            player_card_history = [[] for _ in range(env.n_players)]  # 各プレイヤーが出したカードの順番
            
            actions = [None] * env.n_players
            info_sets = [None] * env.n_players
            
            while not done:
   
                # 各プレイヤーが行動を選択
                for i, player in enumerate(env.players):
                    actions[i] = player.get_action(env._get_flat_state())
                # 環境を進める
                state, rewards, done, info = env.step(actions)
                
                #print(state)
                
                if "score_card" in info:
                    score_card_history.append(info["score_card"])
                #else:
                    #print(f"Warning: 'score_cards' not found at iteration {iteration + 1}")
                    
                if "player_cards" in info:
                    for i, card in enumerate(info["player_cards"]):
                        player_card_history[i].append(card)
                
                player_scores[:, iteration] = env.scores
                
                for i, player in enumerate(env.players):
                    # CFRプレイヤーの学習を更新
                    if isinstance(player, CFRPlayer):
                        utility = [rewards[i] if a == actions[i] else 0 for a in range(env.n_cards)]
                        info_set = env.get_information_set(player.player_index)
                        player.update_regret(info_set, actions[i], utility)
                    # DQNプレイヤーの学習を更新
                    elif isinstance(player, DQNPlayer):
                        # 経験をリプレイバッファに保存
                        player.memorize(state, actions[i], rewards[i], state, done)
                        # リプレイバッファからサンプルしてDQNを更新
                        player.replay()

                        # ターゲットネットワークの更新
                        if (iteration + 1) % player.target_update_frequency == 0:
                            player.update_target_network()
                                    
            scores_history[experiment][iteration] = env.scores
            
            ranks = np.argsort(-np.array(env.scores))
            for rank_index, player_index in enumerate(ranks):
                ranks_history[experiment][iteration][player_index] = rank_index + 1
            
            # ラスト10エピソードの履歴を管理
            if len(last_10_episodes) >= 10:
                last_10_episodes.pop(0)
            last_10_episodes.append((score_card_history, player_card_history))

                            
            # 学習状況を出力（例: 100回ごとにスコアを表示）
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Scores = {env.scores}")
                
            
            # 100回ごとに強化学習エージェントのセーブ
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}")
                for i, player in enumerate(players):
                    if isinstance(player, CFRPlayer):
                        player.save(f"saved_models/cfr_agent_player")
                    elif isinstance(player, DQNPlayer):
                        player.save(f"saved_models/dqn_agent_model.pth")
                    
                        
        
        # ラスト10エピソードの結果を表示
        print_last_10_episodes(last_10_episodes)
        
    # 実験の平均獲得得点と平均獲得順位の配列を作成
    mean_scores_history, mean_ranks_history = calculate_mean_scores_and_ranks(scores_history, ranks_history)
    
    # 平均得点の推移を表示
    plot_mean_scores(mean_scores_history)
        
    # 平均順位の推移を表示
    plot_mean_ranks(mean_ranks_history)
    
    # 各エピソードごとの平均１位獲得率の推移を表示
    plot_average_first_place_rate(ranks_history)

    print(scores_history)
    print(ranks_history)
    
    return total_rewards