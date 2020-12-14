import numpy as np
import torch

from agents.deepq import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):
    def learn(self):
        obs_batch, action_batch, new_obs_batch, reward_batch, dones_batch = self.sample_memory()

        indices = np.arrange(self.batch_size)

        obs_action_values = self.network(obs_batch)[indices, action_batch]

        new_obs_action_values = self.target_network(new_obs_batch).max(dim=1)[0]
        new_obs_action_values[dones_batch] = 0.0

        max_actions = torch.argmax(self.network(new_obs_batch), dim=1)

        expected_state_action_values = (new_obs_action_values[indices, max_actions] * self.discount_factor) \
                                       + reward_batch

        loss = self.network.loss(obs_action_values, expected_state_action_values)
        loss.backward()
