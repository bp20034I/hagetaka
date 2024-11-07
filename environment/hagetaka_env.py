# ハゲタカのえじき四人対戦の強化学習用環境

import random
from config import config   





class HagetakaEnv:
    def __init__(self, player_types=None):
        """
        player_types: 各プレイヤーのタイプを指定するリスト。例: ['random', 'stable', 'sta_rand', 'negavoid']
        """
        self.n_players = 4
        self.n_cards = 15
        self.player_cards = [list(range(1, self.n_cards + 1)) for _ in range(self.n_players)]  # 各プレイヤーの持ちカード
        self.score_cards = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(self.score_cards)
        self.current_score_index = 0
        self.scores = [0] * self.n_players  # 各プレイヤーのスコア
        if player_types is None:
            player_types = config['environment']['player_types']
        self.player_types = player_types
        self.past_actions = [[] for _ in range(self.n_players)]  # 各プレイヤーの過去の行動
        self.rand_prb = config['environment']['random_probability']

    def reset(self):
        """環境のリセット"""
        self.player_cards = [list(range(1, self.n_cards + 1)) for _ in range(self.n_players)]  # 手持ちカードのリセット
        random.shuffle(self.score_cards)  # 得点カードをシャッフル
        self.current_score_index = 0
        self.scores = [0] * self.n_players  # スコアのリセット
        self.past_actions = [[] for _ in range(self.n_players)]  # 過去の行動をリセット
        return self._get_flat_state()

    def _get_flat_state(self):
        """状態をフラットな形式に変換"""
        padded_player_cards = [player + [-1] * (self.n_cards - len(player)) for player in self.player_cards]
        flat_state = (
            sum(padded_player_cards, [])  # プレイヤーの手持ちカードを平坦化
            + self.score_cards
            + [self.current_score_index]
            + self.scores
        )
        return flat_state

    def stable_player_action(self, player_index):
        """安定プレイヤの行動選択: 高得点カードなら高いカードを、負の得点カードなら低いカードを選ぶ"""
        available_cards = self.player_cards[player_index]
        score_card = self.score_cards[self.current_score_index]
        if score_card > 0:
            chosen_card = max(available_cards)  # 高得点カードなら最大値を選ぶ
        else:
            chosen_card = min(available_cards)  # 負の得点カードなら最小値を選ぶ
        return chosen_card

    def random_player_action(self, player_index):
        """ランダムプレイヤの行動選択"""
        available_cards = self.player_cards[player_index]
        chosen_card = random.choice(available_cards)  # 手持ちのカードの中からランダムに1枚を選択
        return chosen_card
    
    def stable_or_random_player_action(self, player_index):
        """Choose stable or random"""
        available_cards = self.player_cards[player_index]
        rand_value = random.random()
        if rand_value < self.rand_prb:
            score_card = self.score_cards[self.current_score_index]
            if score_card > 0:
                chosen_card = max(available_cards)
            else:
                chosen_card = min(available_cards)
        else:
            chosen_card = random.choice(available_cards)
        return chosen_card

    def negative_avoidance_player_action(self, player_index):
        """マイナス回避戦略：マイナスカードが出たら持っている一番大きいカードを出す"""
        available_cards = self.player_cards[player_index]
        score_card = self.score_cards[self.current_score_index]
        if score_card < 0:
            chosen_card = max(available_cards)
        else:
            chosen_card = random.choice(available_cards)
        return chosen_card
    
    def step(self, action):
        """ゲームの1ステップを実行"""
        actions = [action]  # エージェントの行動

        # プレイヤー2～4の行動を選択
        for i in range(1, self.n_players):
            if len(self.player_cards[i]) > 0:
                chosen_action = 0
                if self.player_types[i - 1] == 'random':
                    chosen_action = self.random_player_action(i)  # ランダムプレイヤの行動選択
                elif self.player_types[i - 1] == 'stable':
                    chosen_action = self.stable_player_action(i)  # 安定プレイヤの行動選択
                elif self.player_types[i - 1] == 'sta_rand':
                    chosen_action = self.stable_or_random_player_action(i)
                elif self.player_types[i - 1] == 'negavoid':
                    chosen_action = self.negative_avoidance_player_action(i)
                actions.append(chosen_action)
            else:
                actions.append(-1)  # カードが残っていない場合は-1

        score_card = self.score_cards[self.current_score_index]
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
                self.player_cards[i].remove(action)  # 選んだカードを手持ちから削除

        # 各プレイヤーの行動と得点カードをinfoに保存
        info = {
            'score_card': score_card,
            'player_cards': actions
        }
                

        self.current_score_index += 1  # 次の得点カードに進む
        done = self.current_score_index >= len(self.score_cards)  # ゲーム終了のチェック
        state = self._get_flat_state()
        reward = self.scores[0]  # 学習エージェントの報酬（プレイヤー1）
        return state, reward, done, info

    def is_terminal(self):
        """ゲームが終了しているかを返す"""
        return self.current_score_index >= len(self.score_cards)