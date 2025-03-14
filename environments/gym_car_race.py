import os

import gym
import numpy as np
import time


class GymCarRace:
    def __init__(self, max_steps=3000, render=False, motor_error=1, steer_error=1, brake_error=1,
                 action_delay_step=0):
        if render:
            self._env = gym.make('CarRacing-v2', render_mode='human')
        else:
            self._env = gym.make('CarRacing-v2')
        self.max_steps = max_steps
        self.steps = 0

        self.motor_error = motor_error
        self.steer_error = steer_error
        self.brake_error = brake_error
        self.action_queue = []
        self.action_delay_step = action_delay_step

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
        obs_shape = self._env.observation_space.shape
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        return [obs_shape]

    @property
    def get_action_clamp(self):
        return 2

    @property
    def get_action_space_shape(self):
        return self._env.action_space.shape[0]

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        for _ in range(50):
            obs, reward, done, info = self.step([0, 0, 0])
        return obs

    def step(self, a):
        self.steps += 1
        a = np.array(a)

        self.action_queue.append(a)
        if len(self.action_queue) <= self.action_delay_step:
            a = np.zeros(self.get_action_space_shape, dtype=np.float32)
        else:
            a = self.action_queue.pop(0)

        # motor error
        a[0] = a[0] * self.motor_error
        # steer error
        a[1] = a[1] * self.steer_error
        # brake_error
        a[2] = a[2] * self.brake_error

        # a[1] *= 0.2
        # a[0] *= 0.2
        # a[2] *= 1.2

        obs, reward, done, truncation, info = self._env.step(a)
        if self.steps >= self.max_steps:
            done = True
            self.steps = 0

        obs = obs.transpose(2, 0, 1)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None
        return [obs], reward / 100.0, done, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def make_observation(self):
        # shape: [[84, 84, 3], (80), 6]
        obs = []
        obs.append(self.raw_obs)
        return obs

    def close(self):
        self._env.close()


def assert_observation_space_shape(obs, observation_space_shape):
    assert len(obs) == len(
        observation_space_shape), "The length of the observation is not equal to the length of the observation space shape"
    for i in range(len(obs)):
        if isinstance(obs[i], list) or isinstance(obs[i], tuple):
            assert_observation_space_shape(obs[i], observation_space_shape[i])
        else:
            assert list(obs[i].shape) == list(observation_space_shape[
                                                  i]), "The shape of the observation is not equal to the shape of the observation space"


def test_gym_car_race():
    env = GymCarRace()

    obs = env.reset()
    assert_observation_space_shape(obs, env.get_observation_space_shape)
    for i in range(1000):
        # ford = float(input('for'))
        # steer = float(input('steer'))
        i1 = float(input('i1'))
        i2 = float(input('i2'))
        i3 = float(input('i3'))
        obs, reward, done, info = env.step([0, 1, 0])
        # env._env.render()
        assert_observation_space_shape(obs, env.get_observation_space_shape)
        assert isinstance(reward, float), "Step test failed for reward"
        assert isinstance(done, bool), "Step test failed for done"
        if done:
            i = 0
            obs = env.reset()
        if i % 10 == 0:
            from PIL import Image
            # 转置obs
            obs[0] = obs[0].transpose(1, 2, 0)
            image = Image.fromarray(obs[0])
            if not os.path.exists('test_img'):
                os.makedirs('test_img')
            image.save('test_img/gym_obs.png')
            print('reward', reward)
            # break
        # time.sleep(0.1)
    env.close()


if __name__ == "__main__":
    test_gym_car_race()
