import os

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np


#
# class Observation_space:
#     def __init__(self, shape):
#         self.shape = shape


class Car1:
    def __init__(self, file_name, no_graphics, base_port, worker_id, time_scale=40, motor_error=1, steer_error=1,
                 action_delay_step=0):
        self.engine_configuration_channel = EngineConfigurationChannel()
        if file_name == "":
            self.env = UnityEnvironment(worker_id=worker_id,
                                        side_channels=[self.engine_configuration_channel])
        else:
            self.env = UnityEnvironment(file_name=file_name,
                                        worker_id=worker_id,
                                        no_graphics=no_graphics,
                                        side_channels=[self.engine_configuration_channel],
                                        base_port=base_port,
                                        additional_args=['--force-vulkan'])
        self.engine_configuration_channel.set_configuration_parameters(
            width=960,
            height=540,
            # quality_level = 5, #1-5
            time_scale=time_scale  # 1-100, 10执行一轮的时间约为10秒，20执行一轮的时间约为5秒。
            # target_frame_rate = 60, #1-60
            # capture_frame_rate = 60 #default 60
        )
        self.env.reset()

        self.agent_names = self.env.behavior_specs.keys()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.action_type = [self.env.behavior_specs.get(behavior_name).action_spec.is_continuous() for behavior_name in
                            self.agent_names]
        self.observation_space = None

        self.motor_error = motor_error
        self.steer_error = steer_error
        self.action_queue = []
        self.action_delay_step = action_delay_step

    @property
    def get_observation_space(self):
        observation_space_shape = self.get_observation_space_shape
        state_number = sum(np.prod(obs_shape) for obs_shape in observation_space_shape)
        self.observation_space = state_number
        # 把observation_space_shape转换成一个整数，用于构建observation_space
        self.observation_space = int(self.observation_space)
        return self.observation_space

    @property
    def get_observation_space_shape(self):
        agent_name = list(self.agent_names)[0]
        observation_specs = self.env.behavior_specs.get(agent_name).observation_specs
        observation_specs = [i.shape for i in observation_specs]
        observation_specs = [(i[2], i[0], i[1]) if len(i) == 3 else i for i in observation_specs]
        # state_number = sum(np.prod(obs_shape) for obs_shape in observation_specs)
        # self.observation_space = Observation_space(state_number)
        return observation_specs

    def get_action_space(self, agent_index, action_type):
        behavior_specs = self.env.behavior_specs.get(list(self.agent_names)[agent_index])
        return behavior_specs.action_spec.continuous_size if action_type else \
            behavior_specs.action_spec.discrete_branches[0]
        # if action_type:
        #     return behavior_specs.action_spec.continuous_size
        # else:
        #     return behavior_specs.action_spec.discrete_branches[0]

    @property
    def get_action_clamp(self):
        return 1

    @property
    def get_action_space_shape(self):
        action_space_list = [self.get_action_space(agent_index, action_type) for agent_index, action_type in
                             enumerate(self.action_type)]
        return action_space_list[0]

    def reset(self):
        self.env.reset()
        self.env.reset()
        self.env.reset()
        # obs = np.zeros(self.get_observation_space_shape, dtype=np.float32)
        # return obs
        # a = np.zeros(self.action_space)
        # next_state, reward, done, info = self.step(a)
        obs = []
        for obs_spece in self.get_observation_space_shape:
            obs.append(np.zeros(obs_spece, dtype=np.float32))
        return obs

    def step(self, a):
        """
        a: numpy array
        """
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

        action = ActionTuple()
        if len(a.shape) == 1:
            a = np.expand_dims(a, axis=0)
        if self.action_type[0]:
            action.add_continuous(np.asarray(a))
        else:
            action.add_discrete(np.asarray([[a.argmax(-1)]]))
        self.env.set_actions(behavior_name=self.behavior_name, action=action)
        self.env.step()

        next_state = None
        reward = None
        done = None
        info = None
        DecisionSteps, TerminalSteps = self.env.get_steps(self.behavior_name)
        if len(TerminalSteps.reward) == 0:
            obs = []
            for o_index, o in enumerate(DecisionSteps.obs):
                if len(o.shape) == 4:
                    obs.append(np.transpose(o[0], (2, 0, 1)))
                else:
                    obs.append(o[0])
            next_state = obs
            reward = DecisionSteps.reward[0] + DecisionSteps.group_reward[0]
            done = False
            info = False
        else:
            obs = []
            for o_index, o in enumerate(TerminalSteps.obs):
                if len(o.shape) == 4:
                    obs.append(np.transpose(o[0], (2, 0, 1)))
                else:
                    obs.append(o[0])
            next_state = obs
            reward = TerminalSteps.reward[0] + TerminalSteps.group_reward[0]
            reachmaxstep = TerminalSteps.interrupted
            done = True
            info = reachmaxstep[0]
        # next_state = next_state[0]
        return next_state, reward, done, info

    def render(self):
        pass

    def close(self):
        self.env.close()


def assert_observation_space_shape(obs, observation_space_shape):
    assert len(obs) == len(
        observation_space_shape), "The length of the observation is not equal to the length of the observation space shape"
    for i in range(len(obs)):
        if isinstance(obs[i], list) or isinstance(obs[i], tuple):
            assert_observation_space_shape(obs[i], observation_space_shape[i])
        else:
            assert obs[i].shape == observation_space_shape[
                i], "The shape of the observation is not equal to the shape of the observation space"


def test_car1():
    env = Car1(file_name="BuildUGVRace-OneObstacal/RLEnvironments.exe",
               no_graphics=False,
               base_port=15204,
               worker_id=0,
               time_scale=100,
               motor_error=1,
               steer_error=1,
               action_delay_step=0)

    obs = env.reset()

    assert_observation_space_shape(obs, env.get_observation_space_shape)
    for i in range(100):
        # env._env.render()
        obs, reward, done, info = env.step(np.array([1, 0]))
        assert_observation_space_shape(obs, env.get_observation_space_shape)
        reward = float(reward)
        assert isinstance(reward, float), "Step test failed for reward"
        assert isinstance(done, bool), "Step test failed for done"
        if done:
            i = 0
            obs = env.reset()
        if i % 20 == 0:
            # save visual observation
            img = obs[0].transpose(1, 2, 0)
            # 将img的每一个元素都乘以255，然后转换成整数
            img = (img * 255).astype(np.uint8)
            from PIL import Image
            image = Image.fromarray(img)
            if not os.path.exists('test_img'):
                os.makedirs('test_img')
            image.save('test_img/unity_ugv_obs.jpg')
            break
    env.close()


if __name__ == '__main__':
    test_car1()
    print("All tests passed.")
