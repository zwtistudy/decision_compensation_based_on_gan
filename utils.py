import numpy as np

from environments.Torcs import Torcs
from environments.cartpole_env import CartPole
from environments.gym_car_race import GymCarRace
from environments.pendulum_env import Pendulum
# from environments.minigrid_env import Minigrid
# from environments.poc_memory_env import PocMemoryEnv
# from environments.memory_gym_env import MemoryGymWrapper
from environments.Car1 import Car1


def create_env(config: dict, render: bool = False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        config {dict}: The configuration of the environment.

    Returns:
        {env}: Returns the selected environment instance.
    """
    # if config["type"] == "PocMemoryEnv":
    #     return PocMemoryEnv(glob=False, freeze=True)
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "Pendulum":
        return Pendulum()
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True, realtime_mode=render)
    if config["type"] == "GymCarRace":
        motor_error, steer_error, brake_error, action_delay_step = 1, 1, 1, 0
        if 'motor_error' in config:
            motor_error = config["motor_error"]
        if 'steer_error' in config:
            steer_error = config["steer_error"]
        if 'brake_error' in config:
            brake_error = config["brake_error"]
        if 'action_delay_step' in config:
            action_delay_step = config["action_delay_step"]
        return GymCarRace(config["max_steps"], config["render"], motor_error, steer_error, brake_error,
                          action_delay_step)
    # if config["type"] == "Minigrid":
    #     return Minigrid(env_name = config["name"], realtime_mode = render)
    # if config["type"] == "MemoryGym":
    #     return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode = render)
    if config["type"] == "Car1":
        motor_error, steer_error, action_delay_step = 1, 1, 0
        if 'motor_error' in config:
            motor_error = config["motor_error"]
        if 'steer_error' in config:
            steer_error = config["steer_error"]
        if 'action_delay_step' in config:
            action_delay_step = config["action_delay_step"]
        return Car1(config["file_name"], config["no_graphics"], config["base_port"], config["worker_id"],
                    config["time_scale"], motor_error, steer_error, action_delay_step)
    # if config["type"] == "Torcs":
    #     motor_error, steer_error, action_delay_step = 1, 1, 0
    #     if 'motor_error' in config:
    #         motor_error = config["motor_error"]
    #     if 'steer_error' in config:
    #         steer_error = config["steer_error"]
    #     if 'action_delay_step' in config:
    #         action_delay_step = config["action_delay_step"]
    #     return Torcs(motor_error, steer_error, action_delay_step)


def polynomial_decay(initial: float, final: float, max_decay_steps: int, power: float, current_step: int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.zeros(shape)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            # self.S = self.S + (x - old_mean) * (x - self.mean)
            self.S = (self.n - 1) / self.n / self.n * (x - old_mean) * (x - old_mean) + (self.n - 1) / self.n * self.S

        if self.n > 1:
            self.std = np.sqrt(self.S)
        else:
            self.std = np.zeros_like(x)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
