import numpy as np

class Car():
    def __init__(self, env, x_bins, vel_bins, action_bins):
        self.env = env
        self.x_space = np.linspace(-1.2, 0.6, x_bins)
        self.vel_space = np.linspace(-0.07, 0.07, vel_bins)
        self.actions = list(np.linspace(-1, 1, action_bins))
