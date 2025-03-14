import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=48):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.ReLU(),  # The output of the Actor is a tanh layer to bound the actions
        )

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=48):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        return self.critic(torch.cat([state, action], dim=1))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=48):
        super(ActorCritic, self).__init__()

        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic = Critic(state_dim, action_dim, hidden_size)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state, action)
        return action, value


class Generator:
    def __init__(self, state_dim=2, action_dim=2, hidden_size=48, learning_rate=0.001):
        self.model = ActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.predicted_value = None
        self.state_dim = state_dim

    def generate_params(self):
        state = torch.randn(self.state_dim).unsqueeze(0)
        predicted_params, value = self.model(state)
        self.predicted_value = value
        return predicted_params.detach().numpy()

    def update(self, score):
        reward = torch.tensor(score)

        # 计算actor和critic的损失
        actor_loss = -self.predicted_value
        critic_loss = (reward - self.predicted_value) ** 2

        # 计算总损失
        loss = actor_loss + critic_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), loss.item()


generator = Generator(action_dim=1)


def acquistion_function(x):
    score = -((x - 1.25) ** 2) + 1
    return score.sum()


params = generator.generate_params()
params


score = acquistion_function(params)


loss = generator.update(score)


num_epochs = 1000


from tqdm import tqdm


paramss = []
scores = []
actor_losss = []
critic_losss = []
losss = []
with tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        params = generator.generate_params()
        score = acquistion_function(params)
        actor_loss, critic_loss, loss = generator.update(score)

        paramss.append(params)
        scores.append(score)
        actor_losss.append(actor_loss)
        critic_losss.append(critic_loss)
        losss.append(loss)

        pbar.set_postfix({"loss": loss})
        pbar.update(1)


import numpy as np
from matplotlib import pyplot as plt


if len(paramss[0][0]) == 1:
    paramss = [p[0][0] for p in paramss]
    print(paramss)


plt.plot(paramss)
plt.show()


plt.plot(actor_losss)
plt.show()


plt.plot(critic_losss)
plt.show()


plt.plot(losss)
plt.show()


plt.plot(scores)
plt.show()
