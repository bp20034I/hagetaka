class BasePlayer:
    
    def __init__(self, player_index, n_cards):
        self.player_index = player_index
        self.n_cards = n_cards
        self.initial_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.available_cards = self.initial_cards.copy()

    def get_action(self, info_set):
        """情報セットに基づいて行動を選択"""
        raise NotImplementedError("get_actionメソッドはサブクラスで実装してください")

    def update(self, info_set, action, reward):
        """強化学習プレイヤーは学習を行う"""
        pass
    
    def reset(self):
        self.available_cards = self.initial_cards.copy()
