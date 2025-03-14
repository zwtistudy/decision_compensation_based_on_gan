import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space

        # Observation encoder
        in_features_next_layer = 0
        for i, obs_shape in enumerate(self.observation_space_shape):
            if len(obs_shape) > 1:
                # Case: visual observation is available
                # Visual encoder made of 3 convolutional layers
                self.conv1 = nn.Conv2d(obs_shape[0], 32, 8, 4, )
                self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
                self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
                nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
                nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
                nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
                # Compute output size of convolutional layers
                conv_out_size = self.get_conv_output(obs_shape)
                # in_features_next_layer += conv_out_size
                # 通过2层全连接层来处理visual observation，降维到64
                self.vis_fc1 = nn.Linear(conv_out_size, 128)
                self.vis_fc2 = nn.Linear(128, 64)
                nn.init.orthogonal_(self.vis_fc1.weight, np.sqrt(2))
                nn.init.orthogonal_(self.vis_fc2.weight, np.sqrt(2))
                in_features_next_layer += 64
            elif obs_shape[0] > 50:
                # Case: ray observation is available
                # 通过2层卷积层来处理ray observation
                self.ray_conv1 = nn.Conv1d(1, 16, 8, 4)
                self.ray_conv2 = nn.Conv1d(16, 32, 4, 2)
                nn.init.orthogonal_(self.ray_conv1.weight, np.sqrt(2))
                nn.init.orthogonal_(self.ray_conv2.weight, np.sqrt(2))
                conv1_shape = self.conv1d_output_size(obs_shape[0], kernel_size=8, stride=4)
                conv1_shape = self.conv1d_output_size(conv1_shape, kernel_size=4, stride=2)
                conv1_shape *= 32
                # 用一层全连接层来处理ray observation，降维到64
                self.ray_fc1 = nn.Linear(conv1_shape, 64)
                nn.init.orthogonal_(self.ray_fc1.weight, np.sqrt(2))
                in_features_next_layer += 64
            else:
                # Case: vector observation is available
                in_features_next_layer += obs_shape[0]

        # Recurrent layer (GRU or LSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"],
                                          batch_first=True)
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"],
                                           batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        # self.policy_branches = nn.ModuleList()
        self.actor_branch = nn.Linear(in_features=self.hidden_size, out_features=action_space_shape)
        nn.init.orthogonal_(self.actor_branch.weight, np.sqrt(0.01))
        # self.policy_branches.append(actor_branch)
        # self.actor_branch_std = nn.Linear(in_features=self.hidden_size, out_features=action_space_shape)
        # nn.init.orthogonal_(self.actor_branch_std.weight, np.sqrt(0.01))
        self.log_std = nn.Parameter(torch.zeros((1, action_space_shape), dtype=torch.float32))
        # self.policy_branches.append(actor_branch_std)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def conv1d_output_size(self,
            l: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1) -> int:

        from math import floor

        l_out = floor(
            ((l + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )

        return l_out

    def forward(self, obs, recurrent_cell: torch.tensor, device: torch.device, sequence_length: int = 1):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        if isinstance(obs, list):
            batch_size = len(obs[0])
        else:
            batch_size = len(obs)
        # 初始化 batch_size 维的空数组
        h = []
        # Forward observation encoder
        for i, obs_shape in enumerate(self.observation_space_shape):
            # 把h的第i维度的数据取出来
            # h_i = torch.tensor([j[i] for j in obs], dtype=torch.float32, device=device)
            if not isinstance(obs[i], torch.Tensor):
                h_i = torch.tensor(obs[i], dtype=torch.float32, device=device)
            else:
                h_i = obs[i].clone().detach()
            h_i.to(device)
            if len(obs_shape) > 1:
                if len(h_i.shape) == 3:
                    h_i = h_i.unsqueeze(0)
                    batch_size = 1
            #     nparray = h_i.cpu().detach().numpy()
            #     nparray = nparray[0]
            #     nparray = nparray.transpose(1, 2, 0)
            #     # 调换红蓝通道
            #     nparray = nparray[:, :, ::-1]
            #     # 保存彩色图像
            #     nparray = (nparray * 255).astype(np.uint8)
            #     cv2.imwrite("rgb.jpg", nparray)
            #     h_i.to(device)

                # # 分别保存RGB三个通道的图像
                # r = nparray[0]
                # g = nparray[1]
                # b = nparray[2]
                # # 保存图像
                # cv2.imwrite('b.png', b)
                # cv2.imwrite('g.png', g)
                # cv2.imwrite('r.png', r)
                # Propagate input through the visual encoder
                h_i = F.relu(self.conv1(h_i))
                h_i = F.relu(self.conv2(h_i))
                h_i = F.relu(self.conv3(h_i))
                # 通过vis_fc1和vis_fc2
                h_i = h_i.reshape((batch_size, -1))
                h_i = F.relu(self.vis_fc1(h_i))
                h_i = F.relu(self.vis_fc2(h_i))
                # Flatten the output of the convolutional layers
            elif obs_shape[0] > 50:
                # ray observation
                # 使用ray_conv1和ray_conv2
                # 在第1维升维
                h_i = h_i.unsqueeze(1)
                h_i = F.relu(self.ray_conv1(h_i))
                h_i = F.relu(self.ray_conv2(h_i))
                # 使用ray_fc1和ray_fc2
                h_i = h_i.reshape((batch_size, -1))
                h_i = F.leaky_relu(self.ray_fc1(h_i))
            h_i = h_i.reshape((batch_size, -1))
            h.append(h_i)
        # Concatenate all hidden layers
        h = torch.cat(h, dim=1)

        # Forward reccurent layer (GRU or LSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1)  # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent layer
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # The output of the recurrent layer is not activated as it already utilizes its own activations.

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        mean = torch.tanh(self.actor_branch(h_policy))
        # std = F.softplus(self.actor_branch_std(h_policy)) + 1e-5
        std = self.log_std.exp()
        std = std.expand_as(mean)
        # pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return mean, std, value, recurrent_cell

    def get_conv_output(self, shape: tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32,
                          device=device).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32,
                              device=device).unsqueeze(0)
        return hxs, cxs
