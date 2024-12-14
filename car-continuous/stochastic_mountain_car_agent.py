import numpy as np
import random
from tqdm import tqdm

class StochasticMountainCarAgent():
    def __init__(self, car_model, alpha, gamma, log_sample_size):
        self.car_model = car_model
        self.Qtable = np.zeros((
            len(car_model.x_space) + 1,
            len(car_model.vel_space) + 1,
            len(car_model.actions)
        ))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.log_sample_size = log_sample_size  # subset size (O(log(n)))

    def get_sample_action(self):
        return random.choice(self.car_model.actions)

    def stochastic_maximization(self, state, Q, memory=None):
        # randomly sample a subset of actions
        subset_size = min(self.log_sample_size, len(self.car_model.actions))
        sampled_actions = random.sample(self.car_model.actions, subset_size)
        
        if memory is not None:
            sampled_actions += memory
            sampled_actions = list(set(sampled_actions))

        state_idx = tuple(state)
        sampled_indices = [self.car_model.actions.index(a) for a in sampled_actions]
        action_values = [Q[state_idx][a_idx] for a_idx in sampled_indices]

        # Select the best action in the subset
        best_idx = np.argmax(action_values)
        return sampled_actions[best_idx], max(action_values)

    def epsilon_greedy_policy(self, state, Q, epsilon=0.1, memory=None):
        explore = np.random.binomial(1, epsilon)
        if explore:
            return self.get_sample_action()
        else:
            return self.stochastic_maximization(state, Q, memory=memory)[0]

    def get_state(self, obs):
        x, vel = obs
        x_bin = np.digitize(x, self.car_model.x_space)
        vel_bin = np.digitize(vel, self.car_model.vel_space)
        return x_bin, vel_bin

    def train(self, num_k_episodes, epsilon):
        all_rewards = []
        memory = {}
        with tqdm(total=num_k_episodes, desc="Training Progress", unit="episode") as pbar:
            for episode in range(num_k_episodes):
                obs, _ = self.car_model.env.reset()
                done = False
                total_reward = 0
                state = self.get_state(obs)
                
                while not done:
                    # Choose action
                    memory_state = memory.get(state, [])
                    action = self.epsilon_greedy_policy(state, self.Qtable, epsilon, memory=memory_state)
                    action_idx = self.car_model.actions.index(action)

                    # Execute action
                    real_action = np.array([action])
                    obs, reward, done, _, _ = self.car_model.env.step(real_action)

                    # Get next state
                    next_state = self.get_state(obs)
                    memory.setdefault(next_state, []).append(action)

                    # stochastic Q-learning update
                    next_best_action, next_best_value = self.stochastic_maximization(next_state, self.Qtable, memory=memory[next_state])
                    td_target = reward + self.gamma * next_best_value
                    td_error = td_target - self.Qtable[state][action_idx]
                    self.Qtable[state][action_idx] += self.alpha * td_error

                    state = next_state
                    total_reward += reward

                all_rewards.append(total_reward)
                pbar.update(1)
                pbar.set_postfix({"Episode Reward": total_reward})
        return np.mean(all_rewards)

    def test(self, num_l_episodes):
        all_rewards = []
        for _ in range(num_l_episodes):
            obs, _ = self.car_model.env.reset()
            done = False
            total_reward = 0
            state = self.get_state(obs)

            while not done:
                action = self.stochastic_maximization(state, self.Qtable)[0]
                action_idx = self.car_model.actions.index(action)
                real_action = np.array([action])
                obs, reward, done, _, _ = self.car_model.env.step(real_action)
                state = self.get_state(obs)
                total_reward += reward
            all_rewards.append(total_reward)
        return np.mean(all_rewards)
