import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from weights.p2 import *


# 定义 Q 网络
class QNetwork(nn.Module):

    def __init__(self, in_size, out_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN 代理
class DQNAgent:

    def __init__(self, env):
        self.env = env
        self.memory = []
        self.losses = []
        self.iter_cnt = 0
        self.epsilon = epsilon_beg
        self.policy_network = QNetwork(env.dim_s, env.dim_a)
        self.critic_network = QNetwork(env.dim_s, env.dim_a)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.load_policy_network()
        self.update_critic_network()

    def update_critic_network(self):
        self.critic_network.load_state_dict(self.policy_network.state_dict())

    def save_policy_network(self):
        torch.save(self.policy_network.state_dict(), weight_path)

    def load_policy_network(self):
        if os.path.exists(weight_path):
            self.policy_network.load_state_dict(torch.load(weight_path))

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def choose_action(self, state, best=False):
        valid_actions = self.env.get_valid_actions()
        if not best and random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            q_values = self.policy_network(state)
            q_values_valid = q_values[valid_actions]
            return valid_actions[int(torch.argmax(q_values_valid).item())]

    def replay(self):
        if len(self.memory) < batch_size:
            return
        self.iter_cnt = (self.iter_cnt + 1) % update_cycle
        batch = self.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        q_values = self.policy_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            q_target = self.critic_network(-next_states)
            q_target = -q_target.max(1).values.unsqueeze(1)
            q_target[dones] = 0.0
            q_target = rewards.unsqueeze(1) + gamma * q_target
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, q_target)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        if self.epsilon > epsilon_end:
            self.epsilon *= epsilon_dec
        if self.iter_cnt % update_cycle == 0:
            self.update_critic_network()
            self.save_policy_network()

    def train(self):
        for episode in range(episodes):
            self.env.reset()
            done = False
            while not done:
                # X
                state = self.env.get_state()
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.env.render()
                self.remember(state, action, reward, next_state, done)
                if not done:
                    # O
                    self.env.exchange_player()
                    state = self.env.get_state()
                    action = self.choose_action(state)
                    next_state, reward, done = self.env.step(action)
                    self.env.render()
                    self.remember(state, action, reward, next_state, done)
                    self.env.exchange_player()
                self.replay()
            print(f"Episode : {episode}")

    def test_with_(self, rival="ai"):
        self.load_policy_network()
        self.env.reset()
        done = False
        while not done:
            state = self.env.get_state()
            action = self.choose_action(state, True)
            _, _, done = self.env.step(action)
            self.env.render()
            if not done:
                self.env.exchange_player()
                if rival == "ai":
                    state = self.env.get_state()
                    action = self.choose_action(state, True)
                else:
                    action = int(input("Input your action: "))
                _, _, done = self.env.step(action)
                self.env.render()
                self.env.exchange_player()

    def show_trend(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.losses, label="Loss")
        plt.title("Loss Curve", fontsize=16)
        plt.xlabel("Iters", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig("loss_curve.png")
        plt.close()
