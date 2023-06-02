import progressbar as pb
import numpy as np
import random
from collections import namedtuple, deque

from model import Policy

import torch
import torch.nn.functional as F
import torch.optim as optim

!pip install progressbar


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_Agent():
    def __init__(self):
        '''Parameters of the PPO Agent'''

        self.discount = 0.995
        self.epsilon = 0.1
        self.beta = 0.01
        self.discount_rate = .99
        self.SGD_epoch = 4

        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
    # usa os métodos globais:
    # - normalize
    # - clipped_surrogate
    # - add_entropy

    def get_policy(self):
        return self.policy

    def step(self, old_probs, states, actions, rewards):
        for _ in range(self.SGD_epoch):

            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate(self.policy, old_probs, states,
                                   actions, rewards, epsilon=self.epsilon, beta=self.beta)

            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L
            # the clipping parameter reduces as time goes on

        self.epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= .995

    def reset(self):
        pass
        # call init() again


def normalize(rewards):
    mean = np.mean(rewards, axis=1)
    std = np.std(rewards, axis=1) + 1.0e-10
    rewards_normalized = (rewards - mean[:, np.newaxis]) / std[:, np.newaxis]
    return rewards_normalized


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):

    # get the discounts using numpy
    discounts = discount ** np.arange(len(rewards))

    rewards_fut = np.asarray(rewards[::-1]).cumsum(axis=0)[::-1]
    rewards_fut_gamma = rewards_fut * discounts[:, np.newaxis]
    rewards_fut_gamma_norm = normalize(rewards_fut_gamma)

    actions = torch.tensor(actions, dtype=torch.int8, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT,
                            new_probs, 1.0-new_probs)

    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    new_probs = torch.tensor(new_probs, dtype=torch.float, device=device)
    R_fut = torch.tensor(rewards_fut_gamma_norm,
                         dtype=torch.float, device=device)

    ratio = new_probs/old_probs

    clipped_obj = torch.min(R_fut * ratio, R_fut *
                            torch.clamp(ratio, 1-epsilon, 1+epsilon))

    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10) +
                (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(clipped_obj + beta*entropy)

    # do i really need all those methods?

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible

    #     def discount_rewards():
    #     pass

    # def normalize():
    #     pass

    # def include_entropy()
    #     pass
    # def step():
    #     pass

    # def act(state):
    #     pass

    # def learn():
    #     pass
