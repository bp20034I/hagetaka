import matplotlib.pyplot as plt
import numpy as np
import random

from environment.hagetaka_env import HagetakaEnv
from config import config

# ラスト10エピソードの得点カードとプレイヤーのカードを表示する関数
def print_last_10_episodes(last_10_episodes):
    print("\nLast 10 Episodes - Score Cards and Player's Cards:")
    for idx, (score_cards, player_cards) in enumerate(last_10_episodes):
        print(f"\nEpisode {idx + 1}:")
        print(f"  Score Cards   : {' '.join(f'{card:>2}' for card in score_cards)}")
        for i, cards in enumerate(player_cards):
            print(f"  Player {i + 1} Cards: {' '.join(f'{card:>2}' for card in cards)}")

# エージェントのトレーニング関数
def train_agent(agent, env, n_episodes, gamma, num_experiments):
    """
    エージェントをトレーニングし、200エピソードごとにプレイヤーごとの平均スコアを計算してグラフ化する関数
    """
    
    player_scores = np.zeros((env.n_players, n_episodes))  # 各プレイヤーのスコアを記録する
    total_rewards = []  # 各エピソードでの累積報酬を保存するリスト
    last_10_episodes = []  # ラスト10エピソードのカード履歴を保存するリスト
    
    # 各プレイヤーの順位スコアの履歴を保存するリスト
    rank_scores_history = [[] for _ in range(env.n_players)]
    
    # ボーナス報酬の設定
    rank_bonuses = [5.0, 2.5, -1.0, -5.0]  # 順位ごとのボーナスとペナルティ    
    
    #累積報酬の記録
    cumulative_rewards = np.zeros((4, n_episodes))
    
    #累積順位の記録
    cumulative_ranks = np.zeros((4, n_episodes))

    
    for experiment in range(config['training']['num_experiments']):
        c_ranks = 0
        print(f"Running experiment {experiment + 1}/{num_experiments}")
        for episode in range(n_episodes):
            state = env.reset()  # 環境のリセット
            agent.reset()  # エピソードごとに使用可能なカードをリセット
            done = False
            total_reward = 0

            score_card_history = []  # 各エピソードの得点カードの順番
            player_card_history = [[] for _ in range(env.n_players)]  # 各プレイヤーが出したカードの順番

            while not done:
                action, log_prob = agent.get_action(state)  # 行動を選択
                next_state, reward, done, info = env.step(action)  # 環境で1ステップを実行

                # 出したカードの情報を保存
                score_card_history.append(info['score_card'])
                for i, card in enumerate(info['player_cards']):
                    player_card_history[i].append(card)

                agent.store_outcome(state, log_prob, reward)  # 結果をエージェントに保存
                state = next_state
                total_reward += reward  # 報酬を累積

            # プレイヤーごとのスコアを記録
            for i in range(env.n_players):
                player_scores[i][episode] = env.scores[i]  # このエピソードでのスコアを保存
                cumulative_rewards[i][episode] += env.scores[i]  # 累積報酬の追加  
            
            # 各プレイヤーの順位スコアを計算
            scores = env.scores
            sorted_indices = np.argsort(scores)[::-1]  # スコア順にソートして順位を取得（降順）
            
            # 順位に基づいたスコアをrank_scores_historyに追加
            for rank, player_index in enumerate(sorted_indices):
                rank_scores_history[player_index].append(rank_bonuses[rank])
                cumulative_ranks[player_index][c_ranks] += rank_bonuses[rank]
            c_ranks += 1
            
            # エージェントの最終ステップの報酬を順位ボーナスで更新
            rank_reward = rank_bonuses[sorted_indices.tolist().index(0)]  # エージェントの順位に対応するボーナス
            agent.rewards[-1] = rank_reward  # エージェントの最終ステップに順位ボーナスを反映        

            agent.update_policy()  # エピソード終了後に方策を更新
            total_rewards.append(total_reward)

            # 進捗を表示
            if (episode + 1) % 100 == 0:
                print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')

            # 200エピソードごとに平均スコアを計算して表示
            if (episode + 1) % 200 == 0:
                print(f'\nAverage score after {episode + 1} episodes:')
                avg_scores = player_scores[:, episode-199:episode].mean(axis=1)
                for i, avg_score in enumerate(avg_scores):
                    print(f"  Player {i + 1}: {avg_score:.2f}")

            # ラスト10エピソードのカード履歴を保存
            if len(last_10_episodes) >= 10:
                last_10_episodes.pop(0)  # 最後の10エピソードのみ保持するため、古いエピソードを削除
            last_10_episodes.append((score_card_history, player_card_history))
            
        # ラスト10エピソードの結果を表示
        print_last_10_episodes(last_10_episodes)

        # 最後に各プレイヤーの平均スコアをグラフで出力
        # plot_average_scores(player_scores, n_episodes)

        # 200エピソードごとの順位スコアの平均をグラフで表示
        # plot_rank_scores(rank_scores_history, n_episodes, env)
        
    # 実験回数で平均をとる
    cumulative_rewards = cumulative_rewards / num_experiments
    cumulative_ranks = cumulative_ranks / num_experiments
    
    # エピソードごとの平均累積報酬をプロットする関数
    plot_mean_rewards(cumulative_rewards, n_episodes)
    
    # エピソードごとの平均順位をプロットする関数
    plot_mean_ranks(cumulative_ranks, n_episodes)

    return total_rewards

# グラフをプロットする関数
def plot_average_scores(player_scores, n_episodes):
    episodes = np.arange(1, n_episodes + 1)
    plt.figure(figsize=(10, 6))

    # 各プレイヤーのスコアの移動平均をプロット
    for i in range(player_scores.shape[0]):
        avg_scores = np.convolve(player_scores[i], np.ones(200)/200, mode='valid')  # 移動平均を計算
        plt.plot(np.arange(200, n_episodes + 1), avg_scores, label=f'Player {i + 1}')

    plt.title('Average Scores of Players Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Score (over last 200 episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig("average_score.png")
    print("Graph saved as 'average_score.png'")
    
    
# 各プレイヤーの順位スコアの推移をグラフで表示する関数
def plot_rank_scores(rank_scores_history, n_episodes, env):
    num_points = n_episodes // 200  # 200エピソードごとにプロットするポイント数
    avg_rank_scores = np.zeros((env.n_players, num_points))

    for i in range(env.n_players):
        for j in range(num_points):
            avg_rank_scores[i][j] = np.mean(rank_scores_history[i][j*200:(j+1)*200])

    plt.figure(figsize=(10, 6))
    for i in range(env.n_players):
        plt.plot(np.arange(1, num_points + 1) * 200, avg_rank_scores[i], label=f'Player {i+1}')
    plt.title('Average Rank Scores of Players Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Rank Score (over last 200 episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("rank_scores_history shape:", len(rank_scores_history), len(rank_scores_history[0]) if rank_scores_history else 0)

    
    plt.savefig("average_rank.png")
    print("Graph saved as 'average_rank.png'")  

"""
    
env = HagetakaEnv(player_types=['random', 'negavoid', 'sta_rand'])  # 環境の初期化
"""
state_size = 80  # 状態の次元数を取得
action_size = 15  # 行動の数（カードの枚数）
"""
agent = REINFORCEAgent(state_size, action_size)  # エージェントの初期化

n_episodes = 100000  # トレーニングするエピソード数

"""
gamma = 0.50 # 割引率

"""
# トレーニングを実行
total_rewards = train_agent(agent, env, n_episodes, gamma)
"""
# エピソードごとの平均累積報酬をプロットする関数
def plot_mean_rewards(cumulative_rewards, n_episodes):
    episodes = np.arange(1, n_episodes + 1)
    plt.figure(figsize=(10, 6))
    for i in range(cumulative_rewards.shape[0]):
        plt.plot(episodes, cumulative_rewards[i], label=f'Player {i+1}')
        
    plt.title('Average Reward of REINFORCE Agent Over Experiments')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig("average_rewards_over_num_experiments.png")
    print("Graph saved as 'average_rewards_over_experiments.png'")
    
# エピソードごとの平均累積報酬をプロットする関数
def plot_mean_ranks(cumulative_ranks, n_episodes):
    episodes = np.arange(1, n_episodes + 1)
    plt.figure(figsize=(10, 6))
    for i in range(cumulative_ranks.shape[0]):
        plt.plot(episodes, cumulative_ranks[i], label=f'Player {i+1}')
        
    plt.title('Average rank of REINFORCE Agent Over Experiments')
    plt.xlabel('Episode')
    plt.ylabel('Average Rank')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig("average_ranks_over_num_experiments.png")
    print("Graph saved as 'average_ranks_over_experiments.png'")