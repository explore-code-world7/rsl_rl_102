import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

"""
Input: [num_env, dim_obs=164]
Global_extractor: [512, 256]
Leg Extractor: [128, 64, 12]
rest Extractor: [128, 64, 15]
hand Extractor: [128, 64, 14]
"""

class Depart_Actor(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_actions,
                        activation,
                        num_leg_actions = 12,
                        num_hand_actions = 14,
                        num_rest_actions = 15,
                        global_dims = [256, 256],
                        leg_dims = [64],
                        rest_dims = [64],
                        hand_dims = [64],
                        **kwargs):

        super(Depart_Actor, self).__init__()
        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        mlp_input_dim_a = num_actor_obs

        leg_layers = []
        rest_layers = []
        hand_layers = []

        leg_layers.append(nn.Linear(self.num_actor_obs, global_dims[0]))
        rest_layers.append(nn.Linear(self.num_actor_obs, global_dims[0]))
        hand_layers.append(nn.Linear(self.num_actor_obs, global_dims[0]))

        for l in range(len(global_dims)-1):
            leg_layers.append(nn.Linear(global_dims[l], global_dims[l + 1]))
            leg_layers.append(activation)
            rest_layers.append(nn.Linear(global_dims[l], global_dims[l + 1]))
            rest_layers.append(activation)
            hand_layers.append(nn.Linear(global_dims[l], global_dims[l + 1]))
            hand_layers.append(activation)

        leg_layers.append(nn.Linear(global_dims[-1], leg_dims[0]))
        leg_layers.append(activation)
        for l in range(len(leg_dims)):
            if l == len(leg_dims) - 1:
                leg_layers.append(nn.Linear(leg_dims[l], num_leg_actions))
            else:
                leg_layers.append(nn.Linear(leg_dims[l], leg_dims[l + 1]))
                leg_layers.append(activation)
        self.leg_extractor = nn.Sequential(*leg_layers)

        rest_layers.append(nn.Linear(global_dims[-1], rest_dims[0]))
        rest_layers.append(activation)
        for l in range(len(rest_dims)):
            if l == len(rest_dims) - 1:
                rest_layers.append(nn.Linear(rest_dims[l], num_rest_actions))
            else:
                rest_layers.append(nn.Linear(rest_dims[l], rest_dims[l + 1]))
                rest_layers.append(activation)
        self.rest_extractor = nn.Sequential(*rest_layers)

        hand_layers.append(nn.Linear(global_dims[-1], hand_dims[0]))
        hand_layers.append(activation)
        for l in range(len(hand_dims)):
            if l == len(hand_dims) - 1:
                hand_layers.append(nn.Linear(hand_dims[l], num_hand_actions))
            else:
                hand_layers.append(nn.Linear(hand_dims[l], hand_dims[l + 1]))
                hand_layers.append(activation)
        self.hand_extractor = nn.Sequential(*hand_layers)

    def forward(self, observations):
        # print(f"In DepartAC.py, observation.shape = {observations.shape}")
        # hidden = self.global_extractor(observations)
        hidden = observations
        leg_mean = self.leg_extractor(hidden)
        rest_mean = self.rest_extractor(hidden)
        hand_mean = self.hand_extractor(hidden)
        # print(f"observations = {observations.shape}, leg_mean = {leg_mean.shape}, hand_mean = {hand_mean.shape}")
        mean = torch.concatenate([leg_mean, rest_mean, hand_mean], dim=-1)
        # return leg_mean, hand_mean
        return mean

    # def leg_forward(self, observations):
    #     hidden = self.global_extractor(observations)
    #     leg_mean = self.leg_extractor(hidden)        
    #     return leg_mean

class AC_Depart(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_leg_actions = 12,
                        num_hand_actions = 14,
                        num_rest_actions = 15,
                        global_dims = [512, 256],
                        leg_dims = [64],
                        hand_dims = [64],
                        rest_dims = [64],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        noise_std_type: str = "scalar",
                        **kwargs):
        if kwargs:
            print("Depart.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(AC_Depart, self).__init__()

        self.num_leg_actions = num_leg_actions
        self.num_hand_actions = num_hand_actions
        self.num_rest_actions = num_rest_actions

        activation = get_activation(activation)

        mlp_input_dim_c = num_critic_obs

        self.actor = Depart_Actor(num_actor_obs, num_actions, activation, num_leg_actions, num_hand_actions, num_rest_actions,\
            global_dims, leg_dims, hand_dims, rest_dims)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # analysis:actor is normal mlp, if mean has nan, the error must exists in observations
        mean = self.actor(observations)
        # print(f"mean.shape = {mean.shape}")
        # mean = torch.concatenate([leg_mean, hand_mean], dim=-1)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, mean*0. + std)

    def update_leg_distribution(self, observations):
        leg_mean = self.actor.leg_forward(observations)
        # mid = list(leg_mean.shape)
        # mid[-1] = self.actor.num_actions - mid[-1]
        # mean = torch.concatenate([leg_mean, torch.zeros(mid).to(observations.device)],dim=-1)
        # self.distribution = Normal(mean, mean*0. + self.std)
        self.distribution = Normal(leg_mean, leg_mean*0. + self.std)

    # append noise at training
    def act(self, observations, **kwargs):
        # 原参数: self, observations, **kwargs
        # if "mere_leg" in kwargs.keys() and kwargs["mere_leg"] == True:
        #     self.update_leg_distribution(observations)
        # else:
        self.update_distribution(observations)

        # mlp从 hidden_size变为action_num
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
        # 与旧的动作概率相减，再exp，用于计算surrogate loss

    # directly output action at inference
    def act_inference(self, observations, mere_leg=False):
        # print(f"observations.shape = {observations.shape}")
        # if mere_leg:
        #     leg_mean = self.actor.leg_forward(observations)
        #     mid = list(leg_mean.shape)
        #     mid[-1] = self.actor.num_actions - mid[-1]
        #     mean = torch.concatenate([leg_mean, torch.zeros(mid).to(observations.device)],dim=-1)
        #     # return leg_mean
        # else:
        mean = self.actor(observations)
        
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
