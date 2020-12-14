import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from agents.deepq import DeepQAgent


class DuellingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, learning_rate):
        super().__init__()

        self.layer_1 = nn.Linear(*input_dims, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 32)

        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, n_actions)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        data = F.relu(self.layer_1(obs))
        data = F.relu(self.layer_2(data))
        data = F.relu(self.layer_3(data))

        value = self.value(data)
        advantage = self.advantage(data)

        return value, advantage


class DuellingDeepQAgent(DeepQAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = DuellingDeepQNetwork(kwargs["input_dims"], kwargs["n_actions"], kwargs["learning_rate"])
        self.target_network = DuellingDeepQNetwork(kwargs["input_dims"], kwargs["n_actions"], kwargs["learning_rate"])

    def epsilon_greedy_choice(self, obs):
        if np.random.random() > self.epsilon:
            return np.random.choice(self.action_space)

        _, advantage = self.network.forward(torch.tensor([obs], dtype=torch.float))
        return torch.argmax(advantage).item()

    def learn(self):
        obs_batch, action_batch, new_obs_batch, reward_batch, dones_batch = self.sample_memory()

        obs_value, obs_advantage = self.network(obs_batch)
        new_obs_value, new_obs_advantage = self.target_network(new_obs_batch)

        indices = np.arange(self.batch_size)

        obs_action_values = \
            torch.add(obs_value, (obs_advantage - obs_advantage.mean(dim=1, keepdim=True)))[indices, action_batch]

        new_obs_action_values = \
            torch.add(new_obs_value, (new_obs_advantage - new_obs_advantage.mean(dim=1, keepdim=True))).max(dim=1)[0]
        new_obs_action_values[dones_batch] = 0.0

        expected_state_action_values = (new_obs_action_values * self.discount_factor) + reward_batch

        loss = self.network.loss(obs_action_values, expected_state_action_values)
        loss.backward()
