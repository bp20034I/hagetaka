import os
import numpy as np
from collections import defaultdict
import logging
import dill

from agent.base_player import BasePlayer

# ログ設定
logging.basicConfig(
    filename="cfr_player.log",  # ログファイルのパス
    level=logging.INFO,         # ログレベルをINFOに設定
    format="%(asctime)s - %(levelname)s - %(message)s"
)
class CFRPlayer(BasePlayer):
    def __init__(self, player_index, n_cards, n_players):
        self.player_index = player_index
        self.n_actions = 15
        self.n_cards = n_cards
        self.n_players = n_players
        self.initial_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.available_cards = self.initial_cards.copy()
        self.regret_sum = defaultdict(lambda: np.zeros(n_cards))
        self.strategy_sum = defaultdict(lambda: np.zeros(n_cards))
        self.estimated_strategies = [defaultdict(lambda: np.ones(n_cards) / n_cards) for _ in range(n_players)]
    
    def reset(self):
        self.available_cards = self.initial_cards.copy()
    
    def get_strategy(self, info_set):
        """現在の情報セットに基づく戦略を取得"""
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = np.zeros(self.n_cards)
        regret = np.maximum(self.regret_sum[info_set], 0)
        normalizing_sum = np.sum(regret)
        if normalizing_sum > 0:
            strategy = regret / normalizing_sum
        else:
            strategy = np.ones(self.n_cards) / self.n_cards
            
        self.strategy_sum[info_set] += strategy
        return strategy

    def get_action(self, info_set):
        """現在の戦略に基づいて行動を選択"""
        
        strategy = self.get_strategy(info_set)
        action_probabilities = [strategy[i - 1] if i in self.available_cards else 0 for i in range(1, self.n_cards + 1)]
        
        action_probabilities = [strategy[self.available_cards.index(i)] for i in self.available_cards]
        
        total_probability = sum(action_probabilities)
        if total_probability > 0:
            action_probabilities = [p / total_probability for p in action_probabilities]
        else:
            action_probabilities = [1 / len(self.available_cards) for _ in self.available_cards]
    
        # デバッグ情報を出力
        #logging.info(f"Debug Info:")
        #logging.info(f"  Available cards: {self.available_cards}")
        #logging.info(f"  Strategy: {strategy}")
        #logging.info(f"  Action probabilities: {action_probabilities}")
        #logging.info(f"  Length of available_cards: {len(self.available_cards)}")
        #logging.info(f"  Length of action_probabilities: {len(action_probabilities)}")

        # エラーチェック
        if len(action_probabilities) != len(self.available_cards):
            logging.error(f"Mismatch detected:")
            logging.error(f"  Action probabilities: {action_probabilities}")
            logging.error(f"  Available cards: {self.available_cards}")
            raise ValueError("Mismatch between available cards and action probabilities.")

        # ランダムに行動を選択
        action = np.random.choice(self.available_cards, p=action_probabilities)
        self.available_cards.remove(action)
        return action

    def update_regret(self, info_set, action, utility):
        """後悔値を更新"""
        strategy = self.get_strategy(info_set)
        node_util = sum(strategy[a] * utility[a] for a in range(self.n_cards))
        for a in range(self.n_cards):
            self.regret_sum[info_set][a] += utility[a] - node_util
        
        #print(f"Updated regret for info_set: {info_set}, strategy: {self.regret_sum[info_set]}")
        
        self.strategy_sum[info_set] += strategy
        #print(f"Strategy for info_set {info_set}: {self.strategy_sum[info_set]}")

    def observe(self, player_index, info_set, strategy):
        """他プレイヤーの戦略を観測し推定"""
        self.estimated_strategies[player_index][info_set] = strateg

    def predict_action(self, player_index, info_set):
        """推定された戦略に基づいて他プレイヤーの行動を予測"""
        strategy = self.estimated_strategies[player_index][info_set]
        action = np.random.choice(range(self.n_cards), p=strategy)
        return action

    def save(self, filepath):
            """エージェントの状態をファイルに保存"""
            
            if not filepath.endswith(".dill"):
                filepath += ".dill"
            
            data = {
                "strategy_sum": self.strategy_sum,
                "regret_sum": self.regret_sum,
                "player_index": self.player_index,
                "n_cards": self.n_cards,
                "n_players": self.n_players,
            }
            
            #print("Saving data:", data)
            
            with open(filepath, "wb") as f:
                dill.dump(data, f)
            print(f"Agent state saved to {filepath}")
            
    @classmethod
    def load(cls, filepath):
        """ファイルからエージェントの状態をロード"""
        
        if not filepath.endswith(".dill"):
            filepath += ".dill"
        
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist. Creating a new agent.")
            return None
        try:
            with open(filepath, "rb") as f:
                data = dill.load(f)
        except (EOFError, dill.UnpicklingError) as e:
            print(f"Failed to load file {filepath}: {e}. Skipping load.")
            return None
        except Exception as e:
            print(f"Unexpected error while loading {filepath}: {e}. Skipping load.")
            return None
        # print("Loaded data:", data)
        
        agent = cls(
            player_index=data["player_index"],
            n_cards=data["n_cards"],
            n_players=data["n_players"]
        )
        agent.strategy_sum = data["strategy_sum"]
        agent.regret_sum = data["regret_sum"]
        print(f"Agent state loaded from {filepath}")
        return agent
