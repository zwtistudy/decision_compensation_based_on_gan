import gymnasium as gym
import numpy as np
import time

class Pendulum:
    def __init__(self):
        self._env = gym.make("Pendulum-v1")

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def get_observation_space(self):
        observation_space_shape = self._env.observation_space.shape
        observation_space = np.prod(observation_space_shape)
        observation_space = int(observation_space)
        return observation_space

    @property
    def get_observation_space_shape(self):
        return [self._env.observation_space.shape]

    @property
    def get_action_clamp(self):
        return 2

    @property
    def get_action_space_shape(self):
        return self._env.action_space.shape[0]

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, done, truncation, info = self._env.step(action[0])
        self._rewards.append(reward)
        if done or truncation:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, reward / 100.0, done or truncation, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()