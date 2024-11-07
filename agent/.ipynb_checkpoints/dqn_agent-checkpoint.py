import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment.hagetaka_env import HagetakaEnv

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
        x = self.fc3(x)  # 出力はQ値（softmaxは不要）
        return x

# DQNエージェント
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.policy_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # ターゲットネットワークの初期化
        self.update_target_network()

    def update_target_network(self):
        """ターゲットネットワークの更新"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        """経験リプレイ用メモリに保存"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """ε-greedy法で行動を選択"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state)
        return q_values.argmax().item()

    def replay(self):
        """経験リプレイで学習"""
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

# DQNエージェントのトレーニング関数
def train_dqn_agent(agent, env, n_episodes, target_update_freq=10):
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)  # 行動を選択
            next_state, reward, done, _ = env.step(action)  # 環境で1ステップ実行
            agent.memorize(state, action, reward, next_state, done)  # 経験をメモリに保存
            agent.replay()  # 経験リプレイで学習
            state = next_state
            total_reward += reward

        # 進捗を表示
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')

        # ターゲットネットワークの更新
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

    print("Training complete.")

# 使用例
state_size = 80  # 状態の次元数
action_size = 15  # 行動の数（カードの枚数）
env = HagetakaEnv(player_types=['random', 'negavoid', 'sta_rand'])
agent = DQNAgent(state_size, action_size)

n_episodes = 1000  # トレーニングするエピソード数
train_dqn_agent(agent, env, n_episodes)
