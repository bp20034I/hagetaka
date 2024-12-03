import random

from agent.base_player import BasePlayer
from config import config

pb = config['environment']['random_probability']

class RandomPlayer(BasePlayer):
    def get_action(self, info_set):
        action = random.choice(self.available_cards)
        self.available_cards.remove(action)
        return action

class StablePlayer(BasePlayer):
    def get_action(self, info_set):
        score_card = info_set[-1]
        if score_card > 0:
            action = max(self.available_cards)
        else:
            action = min(self.available_cards)
        self.available_cards.remove(action)
        return action
    
class StableRandomPlayer(BasePlayer):
    def get_action(self, info_set):
        rand_value = random.random()
        if rand_value < pb:
            score_card = info_set[-1]
            if score_card > 0:
                action = max(self.available_cards)
            else:
                action = min(self.available_cards)
        else:
            action = random.choice(self.available_cards)
        self.available_cards.remove(action)
        return action
    
class NegativePlayer(BasePlayer):
    def get_action(self, info_set):
        score_card = info_set[-1]
        if score_card < 0:
            action = self.available_cards[len(self.available_cards) // 2]
        else:
            action = random.choice(self.available_cards)
        self.available_cards.remove(action)
        return action
    
class MiddlePlayer(BasePlayer):
    def get_action(self, info_set):
        score_card = info_set[-1]
        if score_card > 0 and score_card < 6:
            action = min(self.available_cards)
        else:
            action = random.choice(self.available_cards)
        self.available_cards.remove(action)
        return action
    
class NegativeMiddlePlayer(BasePlayer):
    def get_action(self, info_set):
        score_card = info_set[-1]
        if score_card < 0:
            action = self.available_cards[len(self.available_cards) // 2]
        elif score_card > 0 and score_card < 6:
            action = min(self.available_cards)
        else:
            action = random.choice(self.available_cards)
        self.available_cards.remove(action)
        return action
        