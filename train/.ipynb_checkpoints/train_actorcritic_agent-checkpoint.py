def train_actorcritic_agent(agent, env, n_episodes=1000):
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # エージェントが行動を選択
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # エージェントを更新
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")
        
from environment.hagetaka_env import HagetakaEnv
from config import config
import matplotlib.pyplot as plt
import numpy as np

# ラスト10エピソードの得点カードとプレイヤーのカードを表示する関数
def print_last_10_episodes(last_10_episodes):
    print("\nLast 10 Episodes - Score Cards and Player's Cards:")
    for idx, (score_cards, player_cards) in enumerate(last_10_episodes):
        print(f"\nEpisode {idx + 1}:")
        print(f"  Score Cards   : {' '.join(f'{card:>2}' for card in score_cards)}")
        for i, cards in enumerate(player_cards):
            print(f"  Player {i + 1} Cards: {' '.join(f'{card:>2}' for card in cards)}")

# DQNエージェントのトレーニング関数
def train_actorcritic_agent(agent, env, n_episodes=1000):
    """
    DQNエージェントをトレーニングし、200エピソードごとにプレイヤーごとの平均スコアを計算してグラフ化する関数
    """
    
    player_scores = np.zeros((env.n_players, n_episodes))  # 各プレイヤーのスコアを記録する
    total_rewards = []  # 各エピソードでの累積報酬を保存するリスト
    last_10_episodes = []  # ラスト10エピソードのカード履歴を保存するリスト
    cumulative_rewards = np.zeros((env.n_players, n_episodes))  # 累積報酬の記録

    for experiment in range(num_experiments):
        print(f"Running experiment {experiment + 1}/{num_experiments}")
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                # エージェントが行動を選択
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                # エージェントを更新
                agent.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

            # プレイヤーごとのスコアを記録
            for i in range(env.n_players):
                player_scores[i][episode] = env.scores[i]
                cumulative_rewards[i][episode] += env.scores[i]  # 累積報酬の追加

            total_rewards.append(total_reward)
            
            # 進捗を表示
            if (episode + 1) % 100 == 0:
                print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')

            # 200エピソードごとの平均スコアを計算して表示
            """
            if (episode + 1) % 200 == 0:
                print(f'\nAverage score after {episode + 1} episodes:')
                avg_scores = player_scores[:, episode-199:episode].mean(axis=1)
                for i, avg_score in enumerate(avg_scores):
                    print(f"  Player {i + 1}: {avg_score:.2f}")
                    """

            # ラスト10エピソードのカード履歴を保存
            if len(last_10_episodes) >= 10:
                last_10_episodes.pop(0)  # 最後の10エピソードのみ保持するため、古いエピソードを削除
            last_10_episodes.append((score_card_history, player_card_history))
        
        # ラスト10エピソードの結果を表示
        print_last_10_episodes(last_10_episodes)

    # 実験回数で平均をとる
    cumulative_rewards = cumulative_rewards / num_experiments
    
    # エピソードごとの平均累積報酬をプロットする関数
    plot_mean_rewards(cumulative_rewards, n_episodes)

    return total_rewards

# エピソードごとの平均累積報酬をプロットする関数
def plot_mean_rewards(cumulative_rewards, n_episodes):
    episodes = np.arange(1, n_episodes + 1)
    plt.figure(figsize=(10, 6))
    for i in range(cumulative_rewards.shape[0]):
        plt.plot(episodes, cumulative_rewards[i], label=f'Player {i+1}')
        
    plt.title('Average Reward of DQN Agent Over Experiments')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig("average_rewards_dqn.png")
    print("Graph saved as 'average_rewards_dqn.png'")
