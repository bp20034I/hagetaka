import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.hagetaka_env import HagetakaEnv  # 環境をインポート

class ActorCriticAgent:
    def __init__(self, state_size, action_size, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Actor: ポリシーネットワーク
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic: 状態価値ネットワーク
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # オプティマイザ
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        """
        状態に基づいて行動を選択
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state).squeeze(0).detach().numpy()
        action = np.random.choice(self.action_size, p=action_probs)
        return action + 1  # カード番号（1〜15）を返す

    def update(self, state, action, reward, next_state, done):
        """
        ActorとCriticを更新
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Critic: 状態価値Vを計算
        value = self.critic(state)
        next_value = self.critic(next_state)

        # TDターゲットとTDエラー
        target = reward + self.gamma * next_value * (1 - done)
        td_error = target - value

        # Criticの損失
        critic_loss = td_error.pow(2).mean()

        # Actorの損失
        action_prob = self.actor(state).squeeze(0)[action - 1]  # 0始まりのインデックス
        actor_loss = -torch.log(action_prob) * td_error.detach()

        # ActorとCriticを同時に更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
