import gym
from tqdm import trange

from agents.<agent> import <agent_class> as Agent
from agents.utils.graphs import plot_score_graph

env = gym.make("CartPole-v1")
agent = Agent(episodes=4000, input_dims=env.observation_space.shape, n_actions=env.action_space.n,
              learning_rate=0.0001, replace_interval=1000, batch_size=32,
              discount_factor=0.98, epsilon=(0.75, 0.35, 0.01), replay_len=1000000)

scores = [0]
for i in (bar := trange(4000)):
    obs = env.reset()
    done = False
    score = 0

    agent.decrement_epsilon()
    bar.set_postfix_str(f"Epsilon: {agent.epsilon}, Score: {scores[-1]}")

    while not done:
        action = agent.epsilon_greedy_choice(obs)
        new_obs, reward, done, _ = env.step(action)

        score += reward

        agent.replay_memory.append((obs, action, new_obs, reward, done))
        agent.learn_step()

        obs = new_obs

    scores.append(score)

plot_score_graph(scores, group=10)
