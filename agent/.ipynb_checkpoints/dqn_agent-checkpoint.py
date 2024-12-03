import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from config import config
from agent.base_player import BasePlayer

# Q-ネットワークを定義
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 出力はQ値
        return x


class DQNPlayer(BasePlayer):
    def __init__(self, player_index, n_cards, n_players):
        self.player_index = player_index
        self.n_cards = n_cards
        self.n_players = n_players
        
        self.learning_rate = config['agent']['learning_rate']
        self.state_size = (n_players * n_cards) + 15 + 1 + n_players
        self.action_size = config['agent']['action_size']
        self.gamma = config['agent']['gamma']
        self.epsilon = config['agent']['epsilon']
        self.epsilon_decay = config['agent']['epsilon_decay']
        self.epsilon_min = config['agent']['epsilon_min']
        self.batch_size = config['agent']['batch_size']
        self.target_update_frequency = config['agent']['target_update_frequency']
        self.memory = deque(maxlen=config['agent']['memory_size'])
        
        self.initial_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.available_cards = self.initial_cards.copy()
        self.policy_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # ターゲットネットワークの初期化
        self.update_target_network()

    def reset(self):
        #エピソードの開始時に使用可能なカードをリセット
        self.available_cards = self.initial_cards.copy()        

    def update_target_network(self):
        #ターゲットネットワークの更新"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        #経験リプレイ用メモリに保存"""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        
        if not self.available_cards:
            raise ValueError("No Available actions left for the agent.")
        
        #ε-greedy法で行動を選択"""
        if random.random() < self.epsilon:
            # 使用可能なカードからランダムに選択
            action = random.choice(self.available_cards)
        else:
            # Qネットワークで行動を選択
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        
            with torch.no_grad():
                q_values = self.policy_network(state_tensor).numpy().flatten()
                
            #print(f"State tensor shape: {state_tensor.shape}")
            #print(f"Q values size: {len(q_values)}")
            #print(state_tensor)

            # 使用可能なカードの中で最大のQ値を持つ行動を選択
            available_q_values = [(i, q_values[i - 1]) for i in self.available_cards]
            action = max(available_q_values, key=lambda x: x[1])[0]

        # 選択したカードを使用済みにする
        self.available_cards.remove(action)
        return action

    def replay(self):
        #経験リプレイで学習"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # アクションのインデックス範囲を制限
        actions = torch.clamp(actions, 0, self.policy_network(states).size(1) - 1)
        
        # Q値の更新
        q_values = self.policy_network(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 損失計算とバックプロパゲーション
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # εの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        #モデルの保存"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'player_index': self.player_index,
            'n_cards': self.n_cards,
            'n_players': self.n_players,
        }, filepath)

    @classmethod
    def load(cls, filepath):
        #モデルの読み込み
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist. Creating a new agent.")
            return None
        checkpoint = torch.load(filepath)
        
        agent = cls(
            player_index=checkpoint["player_index"],
            n_cards=checkpoint["n_cards"],
            n_players=checkpoint["n_players"]
        )
        agent.policy_network.load_state_dict(checkpoint['policy_network'])
        agent.target_network.load_state_dict(checkpoint['target_network'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint['epsilon']
        agent.policy_network.eval()
        agent.target_network.eval()