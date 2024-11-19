import numpy as np
import random
from tqdm import tqdm

class MountainCarAgent():
    def __init__(self, Car_Model, alpha, gamma):
        self.Car_Model = Car_Model
        self.Qtable = np.zeros((len(Car_Model.x_space) + 1, len(Car_Model.vel_space) + 1, len(Car_Model.actions)))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
    
    def get_sample_action(self):
        return random.choice(self.Car_Model.actions)
    
    def epsilon_greedy_policy(self, state, Q, epsilon=0.1):
        explore = np.random.binomial(1, epsilon)
        if explore:
            action = self.get_sample_action()
        else:  # Exploit
            action = self.optimal_policy(state, Q)
        return action

    def optimal_policy(self, state, Q):
        action_idx = np.argmax(Q[state])
        return self.Car_Model.actions[action_idx]

    def get_state(self, obs):
        x, vel = obs
        x_bin = np.digitize(x, self.Car_Model.x_space)
        vel_bin = np.digitize(vel, self.Car_Model.vel_space)
        return x_bin, vel_bin

    def train(self, num_k_episodes, epsilon):
        all_rewards = []
        with tqdm(total=num_k_episodes, desc="Training Progress", unit="episode") as pbar:
            for episode in range(num_k_episodes):
                obs, _ = self.Car_Model.env.reset()
                done = False
                total_reward = 0
                state = self.get_state(obs)
                
                while not done:  # Inner loop for one episode
                    # Choose action
                    action = self.epsilon_greedy_policy(state, self.Qtable, epsilon)
                    action_idx = self.Car_Model.actions.index(action)

                    # Execute action
                    real_action = np.array([action])
                    obs, reward, done, _, _ = self.Car_Model.env.step(real_action)

                    # Get next state
                    next_state = self.get_state(obs)

                    # Update Q-table
                    self.Qtable[state][action_idx] += self.alpha * (
                        reward + self.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state][action_idx]
                    )
                    state = next_state
                    total_reward += reward
                
                all_rewards.append(total_reward)
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Episode Reward": total_reward})
        return np.mean(all_rewards)


    def test(self, num_l_episodes):
        all_rewards = []
        for _ in range(num_l_episodes):
            obs, _ = self.Car_Model.env.reset()
            done = False
            total_reward = 0
            state = self.get_state(obs)

            while not done:
                action = self.optimal_policy(state, self.Qtable)
                action_idx = self.Car_Model.actions.index(action)
                real_action = np.array([action])
                obs, reward, done, _, _ = self.Car_Model.env.step(real_action)
                state = self.get_state(obs)
                total_reward += reward
            all_rewards.append(total_reward)
        return np.mean(all_rewards)
