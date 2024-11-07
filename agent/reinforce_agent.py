#REINFORCEエージェント

import torch
import torch.optim as optim

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # Use softmax to ensure probabilities sum to 1
        return x

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.action_size = action_size
        self.gamma = 0.99  # 割引率

        self.states = []
        self.actions = []
        self.rewards = []
        self.available_actions = list(range(action_size))  # 使用可能なカードリスト

    def reset(self):
        """エピソードの開始時に使用可能なカードをリセット"""
        self.available_actions = list(range(self.action_size))

    def get_action(self, state):
        """行動を選択し、選択されたカードは使用可能リストから除外"""
        state = torch.FloatTensor(state)
        action_probs = self.policy_network(state)

        # 使用可能なカードだけに対応する確率を抽出
        valid_probs = action_probs[self.available_actions]

        """

        # 確率が0にならないように正規化
        if valid_probs.sum().item() == 0 or torch.isnan(valid_probs).any():
            valid_probs = torch.ones_like(valid_probs) / len(valid_probs)
        else:
            epsilon = 1e-9
            valid_probs = valid_probs / (valid_probs.sum() + epsilon)

        action_dist = torch.distributions.Categorical(valid_probs)
        action_idx = action_dist.sample()



        # 確率が0にならないように正規化 + detachを削除
        # if valid_probs.sum().item() == 0 or torch.isnan(valid_probs).any():
        #    valid_probs = torch.ones_like(valid_probs) / len(valid_probs)
        # else:
        epsilon = 1e-9  # Add small epsilon for numerical stability
        valid_probs = valid_probs / (valid_probs.sum() + epsilon)

        action_dist = torch.distributions.Categorical(valid_probs)
        action_idx = action_dist.sample()

        """

   
        valid_probs = valid_probs / valid_probs.sum() 
        valid_probs[torch.isnan(valid_probs)] = 1 / len(valid_probs)  

        action_dist = torch.distributions.Categorical(valid_probs)
        action_idx = action_dist.sample()


        # 対応する実際のカードを取得（ここで +1 して1〜15の範囲に変換）
        action = self.available_actions[action_idx.item()] + 1

        # 選択されたカードを使用済みにする（リストから削除）
        self.available_actions.remove(action - 1)  # available_actionsは0〜14の範囲で管理

        return action, action_dist.log_prob(action_idx)


    def store_outcome(self, state, log_prob, reward):
        self.states.append(state)
        self.actions.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        """方策の更新"""
        R = 0
        rewards_discounted = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            rewards_discounted.insert(0, R)

        # 報酬を標準化
        rewards_discounted = torch.FloatTensor(rewards_discounted)
        rewards_discounted = (rewards_discounted - rewards_discounted.mean()) / (rewards_discounted.std() + 1e-9)

        # 損失関数を計算して方策を更新
        loss = 0
        for log_prob, reward in zip(self.actions, rewards_discounted):
            loss -= log_prob * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # メモリをリセット
        self.states = []
        self.actions = []
        self.rewards = []