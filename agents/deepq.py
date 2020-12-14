import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.utils.replay_buffer import ReplayBuffer


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, learning_rate):
        super().__init__()

        self.layer_1 = nn.Linear(*input_dims, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 32)
        self.layer_output = nn.Linear(32, n_actions)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        data = F.relu(self.layer_1(obs))
        data = F.relu(self.layer_2(data))
        data = F.relu(self.layer_3(data))
        actions = self.layer_output(data)
        return actions


class DeepQAgent:
    def __init__(self, episodes, input_dims, n_actions,
                 learning_rate, replace_interval, batch_size, discount_factor, epsilon, replay_len):
        self.episodes = episodes
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.learning_rate = learning_rate
        self.replace_interval = replace_interval
        self.batch_size = batch_size

        self.discount_factor = discount_factor

        self.epsilon = epsilon[0]
        self.epsilon_initial = epsilon[0]
        self.epsilon_min_by_episode = epsilon[1]
        self.epsilon_min = epsilon[2]

        self.replay_memory = ReplayBuffer(replay_len)

        self.network = DeepQNetwork(input_dims, n_actions, learning_rate)
        self.target_network = DeepQNetwork(input_dims, n_actions, learning_rate)

        self.action_space = [i for i in range(n_actions)]

        self.learn_counter = 0

    def epsilon_greedy_choice(self, obs):
        if np.random.random() > self.epsilon:
            return np.random.choice(self.action_space)

        actions = self.network.forward(torch.tensor([obs], dtype=torch.float))
        return torch.argmax(actions).item()

    def decrement_epsilon(self):
        new_epsilon = self.episodes * ((1 - self.epsilon_min) / (0 - self.epsilon_min_by_episode))
        self.epsilon = max(new_epsilon, self.epsilon_min)

    def update_target_network(self):
        if self.learn_counter % self.replace_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def sample_memory(self):
        batch = list(zip(*self.replay_memory.sample(self.batch_size)))

        obs_batch = torch.tensor(batch[0], dtype=torch.float)
        action_batch = torch.tensor(batch[1], dtype=torch.long)
        new_obs_batch = torch.tensor(batch[2], dtype=torch.float)
        reward_batch = torch.tensor(batch[3])
        dones_batch = torch.tensor(batch[4])

        return obs_batch, action_batch, new_obs_batch, reward_batch, dones_batch

    def learn_step(self):
        if len(self.replay_memory) < self.batch_size:
            return

        self.network.optimizer.zero_grad()

        self.learn_counter += 1
        self.update_target_network()

        self.learn()

        self.network.optimizer.step()

    def learn(self):
        obs_batch, action_batch, new_obs_batch, reward_batch, dones_batch = self.sample_memory()

        obs_action_values = self.network.forward(obs_batch)[np.arange(self.batch_size), action_batch]

        next_obs_action_values = self.target_network.forward(new_obs_batch).max(dim=1)[0]
        next_obs_action_values[dones_batch] = 0.0

        expected_state_action_values = (next_obs_action_values * self.discount_factor) + reward_batch

        loss = self.network.loss(obs_action_values, expected_state_action_values)
        loss.backward()
