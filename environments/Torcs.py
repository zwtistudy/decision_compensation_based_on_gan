from gym_torcs import sample_agent
from gym_torcs import torcs_env

import numpy as np


class Torcs:
    def __init__(self, motor_error=1, steer_error=1, action_delay_step=0):
        self.env = torcs_env.TorcsEnv(vision=False, throttle=False)
        self.reset()
        self.raw_obs = None

        self.motor_error = motor_error
        self.steer_error = steer_error
        self.action_queue = []
        self.action_delay_step = action_delay_step

    @property
    def get_observation_space(self):
        return 72

    @property
    def get_observation_space_shape(self):
        return [(72,)]

    def get_action_space(self, agent_index, action_type):
        return 2

    @property
    def get_action_clamp(self):
        return 1

    @property
    def get_action_space_shape(self):
        return 2

    def reset(self):
        # obs = self.env.reset(relaunch=True)
        while True:
            try:
                self.raw_obs = self.env.reset()
                break
            except:
                print('Torcs reset failed, retrying...')
        obs = self.make_observation()
        return obs

    def step(self, a):
        a = np.array(a)
        # action delay
        self.action_queue.append(a)
        if len(self.action_queue) <= self.action_delay_step:
            a = np.zeros(self.get_action_space_shape, dtype=np.float32)
        else:
            a = self.action_queue.pop(0)

        a = a.clip(-1, 1)

        # motor error
        a[0] = a[0] * self.motor_error
        # steer error
        a[1] = a[1] * self.steer_error

        # only go forward
        a[1] = np.abs(a[1])

        a = [a[1], a[0]]
        self.raw_obs, reward, done, _ = self.env.step(a)
        done = True if done else False

        obs = self.make_observation()
        return obs, reward, done, _

    def render(self):
        pass

    def make_observation(self):
        # Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
        #                                speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
        #                                speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
        #                                speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
        #                                angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
        #                                damage=np.array(raw_obs['damage'], dtype=np.float32),
        #                                opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
        #                                rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
        #                                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
        #                                trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
        #                                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32)/ 100.0,
        #                                lap=np.array( raw_obs["lap"], dtype=np.uint8))
        obs_list = [self.raw_obs.focus, self.raw_obs.speedX, self.raw_obs.speedY, self.raw_obs.speedZ,
                    self.raw_obs.angle, self.raw_obs.damage, self.raw_obs.opponents, self.raw_obs.rpm,
                    self.raw_obs.track, self.raw_obs.trackPos, self.raw_obs.wheelSpinVel, self.raw_obs.lap]
        obs = []
        for o in obs_list:
            if isinstance(o, list) or isinstance(o, tuple):
                obs.extend(o)
            elif isinstance(o, np.ndarray):
                if o.shape == ():
                    obs.append(0)
                else:
                    obs.extend(o)
            else:
                obs.append(o)
        return [np.array(obs, dtype=np.float32)]

    def close(self):
        # self.env.end()
        pass


def assert_observation_space_shape(obs, observation_space_shape):
    if isinstance(observation_space_shape, list) or isinstance(observation_space_shape, tuple):
        for i in range(len(observation_space_shape)):
            assert_observation_space_shape(obs[i], observation_space_shape[i])
    else:
        assert len(obs) == observation_space_shape, "The shape of the observation is not equal to the shape of the observation space"


def test_torcs():
    env = Torcs()

    obs = env.reset()
    assert_observation_space_shape(obs, env.get_observation_space_shape[0])
    total_resard = 0
    for i in range(200):
        # env._env.render()
        obs, reward, done, info = env.step([1, 0])
        assert_observation_space_shape(obs, env.get_observation_space_shape[0])
        assert isinstance(reward, float), "Step test failed for reward"
        assert isinstance(done, bool), "Step test failed for done"
        total_resard += reward
        print('reward:', reward, 'done:', done)
        if done:
            i = 0
            obs = env.reset()
            print('Total reward:', total_resard)
            total_resard = 0
    # env.close()


if __name__ == '__main__':
    test_torcs()
