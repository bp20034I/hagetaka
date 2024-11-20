import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, memory_size, batch_size, target_update_frequency):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.memory = deque(maxlen=memory_size)
        
        self.available_actions = list(range(self.action_size))

        self.policy_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # ターゲットネットワークの初期化
        self.update_target_network()

    def reset(self):
        #エピソードの開始時に使用可能なカードをリセット
        self.available_actions = list(range(self.action_size))        

    def update_target_network(self):
        #ターゲットネットワークの更新"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        #経験リプレイ用メモリに保存"""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        
        if not self.available_actions:
            raise ValueError("No Available actions left for the agent.")
        
        #ε-greedy法で行動を選択"""
        if random.random() < self.epsilon:
            # 使用可能なカードからランダムに選択
            action = random.choice(self.available_actions)
        else:
            # Qネットワークで行動を選択
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_network(state_tensor).numpy().flatten()

            # 使用可能なカードの中で最大のQ値を持つ行動を選択
            available_q_values = [(i, q_values[i]) for i in self.available_actions]
            action = max(available_q_values, key=lambda x: x[1])[0]

        # 選択したカードを使用済みにする
        self.available_actions.remove(action)
        return action + 1

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

        # Q値の更新
        q_values = self.policy_network(states).gather(1, actions - 1).squeeze()
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
        }, filepath)

    def load(self, filepath):
        #モデルの読み込み"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.policy_network.eval()
        self.target_network.eval()