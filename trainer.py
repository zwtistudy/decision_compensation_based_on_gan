import datetime

import numpy as np
import os
import pickle
import torch
import time
from torch import optim
from buffer import Buffer
from model import ActorCriticModel
from worker import Worker
from utils import create_env
from utils import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter


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


class PPOTrainer:
    def __init__(self, config: dict, run_id: str = "run",
                 resume=False, motor_error=1, steer_error=1, action_delay_step=0) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.run_id = run_id
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        # Set device
        print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.device = device

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        # self.writer = SummaryWriter("./summaries/" + run_id + timestamp)
        name = self.run_id + '-' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir='results/' + name)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        self.config["environment"]["worker_id"] = 0
        self.config["environment"]["motor_error"] = motor_error
        self.config["environment"]["steer_error"] = steer_error
        self.config["environment"]["action_delay_step"] = action_delay_step
        # dummy_env = create_env(self.config["environment"])
        # self.observation_space_shape = dummy_env.get_observation_space_shape
        # self.observation_space = dummy_env.get_observation_space
        # self.action_space_shape = dummy_env.get_action_space_shape
        # self.action_clamp = dummy_env.get_action_clamp
        # dummy_env.close()
        # self.observation_space_shape = [(72,)]
        # self.observation_space = 72
        # self.action_space_shape = 2
        # self.action_clamp = 1

        self.observation_space_shape = [(3, 84, 84), (802,), (2,)]
        self.observation_space = 21972
        self.action_space_shape = 2
        self.action_clamp = 1

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, self.observation_space_shape, self.action_space_shape, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, self.observation_space_shape, self.action_space_shape).to(
            self.device)
        latest_model_checkpoint = self._find_latest_model_checkpoint(run_id)
        if resume and latest_model_checkpoint is not None:
            print("Resuming training from checkpoint: {}".format(latest_model_checkpoint))
            self._load_checkpoint(latest_model_checkpoint)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers
        print("Step 4: Init environment workers")
        # self.workers = [Worker(self.config["environment"]) for w in range(self.config["n_workers"])]
        self.workers = []
        for w in range(self.config["n_workers"]):
            self.config["environment"]["worker_id"] = w
            self.workers.append(Worker(self.config["environment"]))

        # Setup observation placeholder
        self.obs = [np.zeros((self.config["n_workers"],) + shap, dtype=np.float32) for shap in
                    self.observation_space_shape]

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            # self.obs[w] = worker.child.recv()
            state = worker.child.recv()
            for i in range(len(self.obs)):
                if i == 2:
                    self.obs[i][w], self.agent_pos = state[i][:2], state[i][2:]
                else:
                    self.obs[i][w] = state[i]

        self.update = 0

        # 归一化
        self.reward_scaling = RewardScaling(shape=1, gamma=self.config["gamma"])
        self.state_img_norm, self.state_ray_norm, self.state_vec_norm = None, None, None
        if not os.path.exists('obs_states'):
            os.mkdir('obs_states')
        img_state_filename = 'obs_states/%s_state_img_norm.pkl' % (run_id, )
        if os.path.exists(img_state_filename):
            self.state_img_norm = pickle.load(open(img_state_filename, 'rb'))
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

    def _find_latest_model_checkpoint(self, run_id: str = None) -> str:
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
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Loading model checkpoint: {}'.format(latest_checkpoint))
        return latest_checkpoint

    def run_training(self) -> None:
        try:
            """Runs the entire training logic from sampling data to optimizing the model."""
            print("Step 6: Starting training")
            # Store episode results for monitoring statistics

            for self.update in range(self.config["updates"]):
                # Decay hyperparameters polynomially based on the provided config
                learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"],
                                                 self.lr_schedule["max_decay_steps"], self.lr_schedule["power"],
                                                 self.update)
                beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], self.update)
                clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"],
                                              self.cr_schedule["max_decay_steps"], self.cr_schedule["power"],
                                              self.update)

                # Sample training data
                reward_mean = self._sample_training_data()
                # reward_avg = self._sample_training_data()
                # self.writer.add_scalar("reward_avg", reward_avg,
                #                        self.update * self.config["n_workers"] * self.config["worker_steps"])

                # Prepare the sampled data inside the buffer (splits data into sequences)
                self.buffer.prepare_batch_dict()

                # Train epochs
                training_stats = self._train_epochs(learning_rate, clip_range, beta)
                training_stats = np.mean(training_stats, axis=0)

                # Store recent episode infos
                # episode_result = self._process_episode_info(episode_infos)

                # Print training statistics
                # result = "{:4} reward={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                #     self.update, sampled_episode_info["reward_mean"],
                #     training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
                result = "step=%d reward=%.2f pi_loss=%.3f v_loss=%.3f entropy=%.3f loss=%.3f value=%.3f advantage=%.3f" % (
                    self.update * self.config["n_workers"] * self.config["worker_steps"],
                    reward_mean if reward_mean else 0, training_stats[0], training_stats[1], training_stats[3],
                    training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
                print(result)

                # Write training statistics to tensorboard
                self._write_training_summary(training_stats, reward_mean)

                # Free memory
                # del(self.buffer.samples_flat)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if self.update % self.config["save_interval"] == 0:
                    # Save the trained model at the end of the training
                    self._save_model()
        except KeyboardInterrupt:
            self._save_model()
            print("Training was interrupted. Model saved.")
            # save state_img_norm, state_ray_norm, state_vec_norm by pickle
            with open('state_img_norm.pkl', 'wb') as f:
                pickle.dump(self.state_img_norm, f)
            with open('state_ray_norm.pkl', 'wb') as f:
                pickle.dump(self.state_ray_norm, f)
            with open('state_vec_norm.pkl', 'wb') as f:
                pickle.dump(self.state_vec_norm, f)
            print("state_img_norm, state_ray_norm, state_vec_norm saved.")

    def _sample_training_data(self):
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        # episode_infos = dict()
        # Sample actions from the model and collect experiences for training
        reward_avg_of_works = np.zeros(self.config["n_workers"])
        reward_of_sequences = []
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Save the initial observations and recurrentl cell states
                # self.buffer.obs[:, t] = torch.tensor(self.obs)
                for i in range(len(self.obs)):
                    self.buffer.obs[i][:, t] = torch.tensor(self.obs[i])
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                    self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                mean, std, value, self.recurrent_cell = self.model(self.obs, self.recurrent_cell, self.device)
                self.buffer.values[:, t] = value

                # Sample actions from each individual policy branch
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                actions = torch.clamp(actions, -self.action_clamp, self.action_clamp)
                log_probs = dist.log_prob(actions)
                self.buffer.actions[:, t] = actions
                self.buffer.log_probs[:, t] = log_probs

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                _ = worker.child.recv()
                obs, reward, done, info = _
                reward_avg_of_works[w] += reward
                reward = self.reward_scaling(reward)
                _new_obs = []
                for o in obs:
                    if len(o.shape) >= 3:
                        _new_obs.append(self.state_img_norm(o))
                    elif o.shape[0] > 50:
                        _new_obs.append(self.state_ray_norm(o))
                    else:
                        _new_obs.append(self.state_vec_norm(o[:2]))
                        self.agent_pos = o[2:]
                obs = _new_obs
                self.buffer.dones[w, t] = done
                self.buffer.rewards[w, t] = reward
                if done:
                    reward_of_sequences.append(reward_avg_of_works[w])
                    reward_avg_of_works[w] = 0
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
                # Store latest observations
                for i in range(len(obs)):
                    if i == 2:
                        self.obs[i][w], self.agent_pos = obs[i][:2], obs[i][2:]
                    else:
                        self.obs[i][w] = obs[i]

        # Calculate advantages
        _, _, last_value, _ = self.model(self.obs, self.recurrent_cell, self.device)
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        # if reward_of_sequences:
        #     reward_mean = np.mean(reward_of_sequences)
        # else:
        #     reward_mean = None
        reward_mean = np.mean(reward_avg_of_works[:len(self.workers)])

        return reward_mean

    def _train_epochs(self, learning_rate: float, clip_range: float, beta: float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
        return train_info

    def _train_mini_batch(self, samples: dict, learning_rate: float, clip_range: float, beta: float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        mean, std, value, _ = self.model(samples["obs"], recurrent_cell, self.device,
                                         self.buffer.actual_sequence_length)

        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        dist_new = torch.distributions.Normal(mean, std)
        log_probs = dist_new.log_prob(samples["actions"])
        # 计算entropy
        entropies = dist_new.entropy()
        # for i, policy_branch in enumerate(policy):
        #     log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
        #     entropies.append(policy_branch.entropy())
        # log_probs = torch.stack(log_probs, dim=1)
        # entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]]

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (
                samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1,
                                                                        self.action_space_shape)  # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy()]

    def _write_training_summary(self, training_stats, reward_mean) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        # if episode_result:
        #     for key in episode_result:
        #         if "std" not in key:
        #             self.writer.add_scalar("episode/" + key, episode_result[key], self.update)
        step = self.update * self.config["n_workers"] * self.config["worker_steps"]
        self.writer.add_scalar("losses/loss", training_stats[2], step)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], step)
        self.writer.add_scalar("losses/value_loss", training_stats[1], step)
        self.writer.add_scalar("losses/entropy", training_stats[3], step)
        self.writer.add_scalar("training/sequence_length", self.buffer.true_sequence_length, step)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), step)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), step)
        if reward_mean:
            self.writer.add_scalar("episode/reward_mean", reward_mean, step)

    @staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self) -> None:
        """
        Use torch to save the model
        """
        if not os.path.exists("./models"):
            os.makedirs("./models")
        dir_name = "./models/%s" % self.run_id
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        filename = "%s-%s.pth" % (self.run_id, self.update * self.config["n_workers"] * self.config["worker_steps"])
        filename = os.path.join(dir_name, filename)
        torch.save(self.model.state_dict(), filename)
        print("Model saved to " + filename)

    def _load_checkpoint(self, model_path: str) -> None:
        """Loads a model checkpoint from the models directory.

        Arguments:
            model_path {str} -- Name of the model file
        """
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded from " + model_path)

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        # exit(0)
