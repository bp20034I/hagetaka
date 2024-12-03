from agent.cfr_agent import CFRPlayer
import dill
import numpy as np
import matplotlib.pyplot as plt    

def calculate_mean_scores_and_ranks(scores_history, ranks_history):
    """
    実験の平均獲得得点と平均獲得順位の配列を作成
    :param scores_history: 各実験、エピソード、プレイヤーの得点 (shape: [num_experiments, n_episodes, n_players])
    :param ranks_history: 各実験、エピソード、プレイヤーの順位 (shape: [num_experiments, n_episodes, n_players])
    :return: 平均得点推移と平均順位推移
    """
    num_experiments, n_episodes, n_players = scores_history.shape

    # 平均得点を計算
    mean_scores_history = np.mean(scores_history, axis=0)  # 実験の次元を平均化

    # 平均順位を計算
    mean_ranks_history = np.mean(ranks_history, axis=0)  # 実験の次元を平均化

    return mean_scores_history, mean_ranks_history

def plot_mean_scores(mean_scores_history):
    """
    平均得点の推移をプロット
    :param mean_scores_history: 平均得点の配列 (shape: [n_episodes, n_players])
    """
    n_episodes, n_players = mean_scores_history.shape
    episodes = np.arange(1, n_episodes + 1)

    plt.figure(figsize=(10, 6))
    for player_index in range(n_players):
        plt.plot(
            episodes,
            mean_scores_history[:, player_index],
            label=f'Player {player_index + 1}'
        )
    plt.title('Average Scores Over Experiments')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("average_scores_over_experiments.png")
    plt.show()
    print("Graph saved as 'average_scores_over_experiments.png'")

def plot_mean_ranks(mean_ranks_history):
    """
    平均順位の推移をプロット
    :param mean_ranks_history: 平均順位の配列 (shape: [n_episodes, n_players])
    """
    n_episodes, n_players = mean_ranks_history.shape
    episodes = np.arange(1, n_episodes + 1)

    plt.figure(figsize=(10, 6))
    for player_index in range(n_players):
        plt.plot(
            episodes,
            mean_ranks_history[:, player_index],
            label=f'Player {player_index + 1}'
        )
    plt.title('Average Ranks Over Experiments')
    plt.xlabel('Episode')
    plt.ylabel('Average Rank')
    plt.gca().invert_yaxis()  # 順位は低いほど良いのでY軸を反転
    plt.legend()
    plt.grid(True)
    plt.savefig("average_ranks_over_experiments.png")
    plt.show()
    print("Graph saved as 'average_ranks_over_experiments.png'")

def plot_average_first_place_rate(ranks_history):
    """
    各エピソードごとの平均1位獲得率の推移をプロット
    :param ranks_history: 各実験、エピソード、プレイヤーの順位 (shape: [num_experiments, n_episodes, n_players])
    """
    num_experiments, n_episodes, n_players = ranks_history.shape

    # 各エピソードごとに1位を獲得した回数を計算
    first_place_counts = (ranks_history == 1).sum(axis=0)  # エピソードごとに1位の回数を集計

    # 平均1位獲得率を計算
    average_first_place_rate = first_place_counts / num_experiments

    # グラフをプロット
    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, n_episodes + 1)
    for player_index in range(n_players):
        plt.plot(
            episodes,
            average_first_place_rate[:, player_index],
            label=f'Player {player_index + 1}'
        )

    plt.title('Average First Place Rate Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average First Place Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig("average_first_place_rate_over_episodes.png")
    plt.show()
    print("Graph saved as 'average_first_place_rate_over_episodes.png'")



# ラスト10エピソードの得点カードとプレイヤーのカードを表示する関数
def print_last_10_episodes(last_10_episodes):
    print("\nLast 10 Episodes - Score Cards and Player's Cards:")
    for idx, (score_cards, player_cards) in enumerate(last_10_episodes):
        print(f"\nEpisode {idx + 1}:")
        print(f"  Score Cards   : {' '.join(f'{card:>2}' for card in score_cards)}")
        for i, cards in enumerate(player_cards):
            print(f"  Player {i + 1} Cards: {' '.join(f'{card:>2}' for card in cards)}")