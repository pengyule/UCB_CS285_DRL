"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers) # unpacks the list of layers into multiple arguments
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False, 
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training 
        self.nn_baseline = nn_baseline  

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        
        # observation is in the shape of (batch_size, ob_dim)
        mean = self.mean_net(observation)
        # mean is in the shape of (batch_size, ac_dim)
        std = torch.exp(self.logstd)
		# std is in the shape of (ac_dim,)
		# Create a normal distribution with the mean and std
        dist = distributions.Normal(mean, std)
        # dist is a distribution object that can be used to sample actions
		# Sample actions from the distribution
		
		# If you want to return the log probability of the action, you can do so
		# log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dimensions
        
        return dist

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        dist = self.forward(observations)
        loss = -dist.log_prob(actions).sum(dim=-1)
        # use loss to step the optimizer
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

# read this file and tell me how nn_baseline could be used in the code
# It could be used to implement a neural network baseline for the policy, which can help reduce variance in policy gradient methods. If `nn_baseline` is set to True, the policy could use a neural network to estimate the value function, which would be subtracted from the rewards to compute advantages during training. This would allow the policy to learn more effectively by focusing on the relative performance of actions rather than their absolute rewards.
# how to provide the baseline neural network to the policy?
# The baseline neural network can be provided to the policy by initializing it as an attribute of the `MLPPolicySL` class. If `nn_baseline` is set to True, you can create a separate neural network for the baseline and use it in the `update` method to compute advantages. You would typically pass the observations through this baseline network to get value estimates, which can then be used to normalize the rewards or compute advantages.
# However, now that in __init__ we didn't create a baseline network, we may assume that the baseline is not used in this policy.
# If you want to use a baseline, you would need to add code to create and manage the baseline network, such as initializing it in the constructor and using it in the `update` method to compute advantages or normalize rewards.