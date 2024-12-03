import random
from environment.hagetaka_env import HagetakaEnv
from config import config

from agent.cfr_agent import CFRPlayer
from agent.dqn_agent import DQNPlayer

from agent.rule_based_players import (
    RandomPlayer,
    StablePlayer,
    StableRandomPlayer,
    NegativePlayer,
    MiddlePlayer,
    NegativeMiddlePlayer,
)

class MultiPlayerHagetakaEnv:
    def __init__(self, n_players, n_cards, player_types):
        """
        ハゲタカのえじきの強化学習用環境
        :param n_players: プレイヤー数
        :param n_cards: 各プレイヤーが持つカードの数
        :param player_types: プレイヤーの戦略タイプのリスト
        """
        super().__init__()
        self.n_players = n_players
        self.n_cards = n_cards
        self.state_size = (n_players * n_cards) + 15 + 1 + n_players
        self.score_cards = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(self.score_cards)
        self.current_score_index = 0
        self.scores = [0] * self.n_players  # 各プレイヤーのスコア
        
        self.players = []
        for i, player_type in enumerate(player_types):
            # ルールベースプレイヤ
            if player_type == "random":
                self.players.append(RandomPlayer(i, n_cards))
            elif player_type == "stable":
                self.players.append(StablePlayer(i, n_cards))
            elif player_type == "sta_rand":
                self.players.append(StableRandomPlayer(i, n_cards))
            elif player_type == "negative":
                self.players.append(NegativePlayer(i, n_cards))
            elif player_type == "middle":
                self.players.append(MiddlePlayer(i, n_cards))
            elif player_type == "negative_middle":
                self.players.append(NegativeMiddlePlayer(i, n_cards))
            # 強化学習エージェント
            elif player_type == "cfr":
                self.players.append(CFRPlayer(i, n_cards, n_players))
            elif player_type == "dqn":
                self.players.append(DQNPlayer(i, n_cards, n_players))
            
            else:
                raise ValueError(f"Unknown player type: {player_type}")

        self.player_cards = [list(range(1, self.n_cards + 1)) for _ in range(self.n_players)]
        self.past_actions = [[] for _ in range(self.n_players)]
        self.rand_prb = config["environment"]["random_probability"]

    def reset(self):
        """環境のリセット"""
        self.player_cards = [list(range(1, self.n_cards + 1)) for _ in range(self.n_players)]
        self.score_cards = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(self.score_cards)
        self.current_score_index = 0
        self.scores = [0] * self.n_players
        self.past_actions = [[] for _ in range(self.n_players)]
        return self._get_flat_state()

    def _get_flat_state(self):
        """状態をフラットな形式に変換"""
        padded_player_cards = [player + [-999] * (self.n_cards - len(player)) for player in self.player_cards]
        
        padded_score_cards = self.score_cards + [-999] * (15 - len(self.score_cards))
        
        # 現在の得点カードを追加
        current_score_card = (
            self.score_cards[self.current_score_index]
            if self.current_score_index < len(self.score_cards)
            else -999
        )
        
        flat_state = (
            sum(padded_player_cards, [])  # プレイヤーの手持ちカードを平坦化
            + padded_score_cards
            + [current_score_card]
            + self.scores
        )
        
        # 必要ならゼロ詰め（状態次元数が不足する場合）
        if len(flat_state) < self.state_size:
            flat_state += [-999] * (self.state_size - len(flat_state))
        
        return flat_state

    def get_information_set(self, player_index):
        """プレイヤー固有の情報セットを取得"""
        player_hand = self.player_cards[player_index]
        
        opponents_hands = []
        for i, hand in enumerate(self.player_cards):
            if i != player_index:
                opponents_hands.extend(hand)
        
        if self.current_score_index >= len(self.score_cards):
            current_score_card = -1
        else:
            current_score_card = self.score_cards[self.current_score_index]
        
        info_set = tuple(player_hand + opponents_hands + [current_score_card])
        
        # print(f"Info set for player {player_index}: {info_set}")
        
        return info_set

    def step(self, actions):
        """
        環境の1ステップを進める
        :param actions: 各プレイヤーが選択した行動のリスト
        :return: 次の状態、報酬、ゲーム終了フラグ、情報
        """
        score_card = self.score_cards[0]
        unique_actions = list(set(actions))

        # 勝者を決定（同じカードを出したプレイヤは無効）
        if len(unique_actions) == 1:
            winner = actions.index(unique_actions[0])
        else:
            action_counts = [actions.count(action) for action in unique_actions]
            valid_actions = [action for action, count in zip(unique_actions, action_counts) if count == 1]

            if not valid_actions:
                winner = None  # 全員無効
            else:
                if score_card > 0:
                    winner = actions.index(max(valid_actions))  # 最大のカードを出したプレイヤが勝ち
                else:
                    winner = actions.index(min(valid_actions))  # 最小のカードを出したプレイヤが勝ち

        # 勝者がいればスコアを追加
        if winner is not None:
            self.scores[winner] += score_card

        # 過去の行動を記録
        for i, action in enumerate(actions):
            if action != -1:
                self.past_actions[i].append(action)

        # 出したカードを手持ちから削除(無効な行動も含む)
        for i, action in enumerate(actions):
            if action != -1 and action in self.player_cards[i]:
                self.player_cards[i].remove(action)

        # 各プレイヤーの行動と得点カードをinfoに保存
        info = {
            "score_card": score_card,
            "player_cards": actions,
        }
        # 得点カードの先頭要素を削除
        self.score_cards.remove(score_card)

        done = self.current_score_index == len(self.score_cards)  # ゲーム終了のチェック
        state = self._get_flat_state()
        rewards = self.scores
        
        return state, rewards, done, info

    def is_terminal(self):
        """ゲームが終了しているかを返す"""
        return self.current_score_index >= len(self.score_cards)

