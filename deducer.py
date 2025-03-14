import datetime

import numpy as np
import os
import pickle
import torch
import time
from torch import optim
# from buffer import Buffer
from tqdm import tqdm

from model import ActorCriticModel
from worker import Worker
from utils import create_env
# from utils import polynomial_decay
# from collections import deque
# from torch.utils.tensorboard import SummaryWriter


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


def _find_latest_model_checkpoint(run_id: str = None) -> str:
    """Finds the latest model checkpoint in the ./checkpoints directory.

    Keyword Arguments:
        run_id {str} -- The run id of the model checkpoint to load. (default: {None})

    Returns:
        str -- The path to the latest model checkpoint.
    """
    models_path = 'models/' + run_id
    if not os.path.exists(models_path):
        return None
    checkpoints = [f for f in os.listdir(models_path) if
                   os.path.isfile(os.path.join(models_path, f)) and f.endswith('.pth')]
    checkpoints = [os.path.join(models_path, f) for f in checkpoints]
    if len(checkpoints) == 0:
        return None
    # latest_checkpoint = max(checkpoints, key=os.path.getctime)
    latest_steps = [i.split('.')[0].split('-')[-1] for i in checkpoints]
    latest_steps = [int(i) for i in latest_steps]
    latest_checkpoint_filename = '%s-%d.pth' % (run_id, max(latest_steps))
    latest_checkpoint = os.path.join(models_path, latest_checkpoint_filename)
    print('Loading model checkpoint: {}'.format(latest_checkpoint))
    return latest_checkpoint




class PPODeducer:
    def __init__(self, config: dict, run_id, device: torch.device = torch.device("cpu"), accelerate=False,
                 motor_error=1, steer_error=1, brake_error=1, action_delay_step=0) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        self.config["environment"]["worker_id"] = 0
        self.config["environment"]["time_scale"] = 20 if accelerate else 1
        self.config["environment"]["motor_error"] = motor_error
        self.config["environment"]["steer_error"] = steer_error
        self.config["environment"]["brake_error"] = brake_error
        self.config["environment"]["action_delay_step"] = action_delay_step
        # Gym CarRacing
        # dummy_env = create_env(self.config["environment"])
        # self.observation_space_shape = dummy_env.get_observation_space_shape
        # self.observation_space = dummy_env.get_observation_space
        # self.action_space_shape = dummy_env.get_action_space_shape
        # self.action_clamp = dummy_env.get_action_clamp
        # dummy_env.close()
        # UGV Race
        self.observation_space_shape = [(3, 84, 84), (802,), (2,)]
        self.observation_space = 21972
        self.action_space_shape = 2
        self.action_clamp = 1
        # # UGV Search
        # self.observation_space_shape = [(3, 84, 84), (802,), (6,)]
        # self.observation_space = 21972
        # self.action_space_shape = 2
        # self.action_clamp = 1
        # Torcs
        # self.observation_space_shape = [(65,)]
        # self.observation_space = 65
        # self.action_space_shape = 2
        # self.action_clamp = 1

        # Initialize the PPO deducer and commence training
        model_path = _find_latest_model_checkpoint(run_id)
        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, self.observation_space_shape, self.action_space_shape).to(
            self.device)
        if model_path is not None:
            self._load_checkpoint(model_path)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers
        print("Step 4: Init environment workers")
        # self.workers = [Worker(self.config["environment"]) for w in range(self.config["n_workers"])]
        self.workers = []
        self.config["n_workers"] = 1
        self.config["environment"]["worker_id"] = 0
        self.workers.append(Worker(self.config["environment"]))

        # Setup observation placeholder
        self.obs = [np.zeros((self.config["n_workers"],) + shap, dtype=np.float32) for shap in
                    self.observation_space_shape]
        self.agent_pos = np.zeros((6,), dtype=np.float32)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        w = 0
        worker = self.workers[0]
        worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        # self.obs[w] = worker.child.recv()
        state = worker.child.recv()
        for i in range(len(self.obs)):
            if i == 2:
                # self.obs[i][w], self.agent_pos = state[i][:2], state[i][2:]
                self.obs[i][w] = state[i]
            else:
                self.obs[i][w] = state[i]
        # print('obs', self.obs[w])

        self.update = 0

        # 归一化
        self.reward_scaling = RewardScaling(shape=1, gamma=self.config["gamma"])
        self.state_img_norm, self.state_ray_norm, self.state_vec_norm = None, None, None
        # if os.path.exists('state_img_norm.pkl'):
        #     self.state_img_norm = pickle.load(open('state_img_norm.pkl', 'rb'))
        # if os.path.exists('state_ray_norm.pkl'):
        #     self.state_ray_norm = pickle.load(open('state_ray_norm.pkl', 'rb'))
        # if os.path.exists('state_vec_norm.pkl'):
        #     self.state_vec_norm = pickle.load(open('state_vec_norm.pkl', 'rb'))
        for shap in self.observation_space_shape:
            if len(shap) >= 3 and self.state_img_norm is None:
                self.state_img_norm = Normalization(shape=shap)
            elif shap[0] > 50 and self.state_ray_norm is None:
                self.state_ray_norm = Normalization(shape=shap[0])
            elif self.state_vec_norm is None:
                self.state_vec_norm = Normalization(shape=shap[0])

    def load_model(self, path: str) -> None:
        """Loads a trained model from a given path.

        Arguments:
            path {str} -- Path to the model to load.
        """
        # pickle.dump((self.model.state_dict(), self.config), open(filename, "wb"))
        model_state_dict, config = pickle.load(open(path, "rb"))
        # state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def _load_checkpoint(self, model_path: str) -> None:
        """Loads a model checkpoint from the models directory.

        Arguments:
            model_path {str} -- Name of the model file
        """
        print('model_path', model_path)
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded from " + model_path)

    def run_deduce(self, episode_num=5) -> float:
        print("Step 6: Starting deducing")
        # Sample actions from the model and collect experiences for training
        w = 0
        reward_mean = 0
        rewards = []
        worker = self.workers[0]
        trojectoties = []
        with tqdm(total=episode_num) as pbar:
            for episode in range(episode_num):
                trojectoty = []
                reward_sum = 0
                while True:
                    # Gradients can be omitted for sampling training data
                    with torch.no_grad():
                        # Save the initial observations and recurrentl cell states

                        # Save the trajectory
                        # trojectoty.append(self.agent_pos)
                        # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                        mean, std, value, self.recurrent_cell = self.model(self.obs, self.recurrent_cell, self.device)

                        # Sample actions from each individual policy branch
                        dist = torch.distributions.Normal(mean, std)
                        actions = dist.sample()
                        actions = torch.clamp(actions, -self.action_clamp, self.action_clamp).cpu().numpy()[0]
                        # steer, motor = actions[:, 0], actions[:, 1]
                        # actions = torch.stack([steer, motor], dim=1).numpy()
                        # actions = actions[0]

                    trojectoty.append(actions)
                    # Send actions to the environments
                    worker.child.send(("step", actions))

                    # Retrieve step results from the environments
                    _ = worker.child.recv()
                    obs, reward, done, info = _
                    _new_obs = []
                    for o in obs:
                        if len(o.shape) >= 3:
                            _new_obs.append(self.state_img_norm(o))
                        elif o.shape[0] > 50:
                            _new_obs.append(self.state_ray_norm(o))
                        else:
                            _new_obs.append(self.state_vec_norm(o))
                            # self.agent_pos = o[2:]
                    obs = _new_obs
                    reward_sum += reward
                    for i in range(len(obs)):
                        self.obs[i][w] = obs[i]
                    if done:
                        # trojectoty.append(self.agent_pos)
                        # Save the trajectory
                        trojectoties.append(trojectoty)
                        # Store the information of the completed episode (e.g. total reward, episode length)
                        # episode_infos.append(info)
                        # Reset agent (potential interface for providing reset parameters)
                        worker.child.send(("reset", None))
                        self.reward_scaling.reset()
                        # Get data from reset
                        obs = worker.child.recv()
                        # Reset recurrent cell states
                        if self.recurrence["reset_hidden_state"]:
                            hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                            if self.recurrence["layer_type"] == "gru":
                                self.recurrent_cell[:, w] = hxs
                            elif self.recurrence["layer_type"] == "lstm":
                                self.recurrent_cell[0][:, w] = hxs
                                self.recurrent_cell[1][:, w] = cxs
                        break

                pbar.set_postfix({
                    'time': time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    'episode': episode,
                    'reward': reward_sum
                })
                pbar.update(1)
                reward_mean += reward_sum
                rewards.append((episode, reward_sum))
        reward_mean /= episode_num
        print("Step 6: Deducing finished, reward_mean: %.2f" % reward_mean)
        return reward_mean, rewards, trojectoties

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        # try:
        #     self.dummy_env.close()
        # except:
        #     pass

        # try:
        #     self.writer.close()
        # except:
        #     pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        # exit(0)
