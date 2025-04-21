# agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple


# Define a simple MLP model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),  # Add batch normalization
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),  # Add batch normalization
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# Define the DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu",
                 gamma=0.99, lr=1e-3, batch_size=64, buffer_size=100_000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000): # larger epsilon_decay = more exploration

        self.device = torch.device(device)
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)
        self.Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def push_transition(self, *args):
        self.memory.append(self.Transition(*args))

    def sample_batch(self):
        return random.sample(self.memory, self.batch_size)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.sample_batch()
        batch = self.Transition(*zip(*batch))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(0, action_batch)

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            expected_q = reward_batch + (1 - done_batch) * self.gamma * next_q

        # Optimize the model
        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
