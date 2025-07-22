import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Goal-conditional Value Function
class GCValueFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_dim=256):
        super().__init__()
    
        input_dim = state_dim + goal_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

        self.l5 = nn.Linear(input_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.l7 = nn.Linear(hidden_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)

        v1 = F.relu(self.l1(x))
        v1 = F.relu(self.l2(v1))
        v1 = F.relu(self.l3(v1))
        v1 = self.l4(v1)

        v2 = F.relu(self.l5(x))
        v2 = F.relu(self.l6(v2))
        v2 = F.relu(self.l7(v2))
        v2 = self.l8(v2)

        return v1, v2
    
    def v_min(self, state, goal):
        v1, v2 = self.forward(state, goal)
        return torch.min(v1, v2)

    def v_1(self, state, goal):
        v1, _ = self.forward(state, goal)
        return v1


# Goal-conditional Q Function
class GCQFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256):
        super().__init__()

        input_dim = state_dim + goal_dim + action_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

        self.l5 = nn.Linear(input_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.l7 = nn.Linear(hidden_dim, hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(x))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)

        return q1, q2

    def q_min(self, state, goal, action):
        q1, q2 = self.forward(state, goal, action)
        return torch.min(q1, q2)

    def q_1(self, state, goal, action):
        q1, _ = self.forward(state, goal, action)
        return q1


# Deterministic Goal-conditional Policy
class Deterministic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()

        self.max_action = max_action

        input_dim = state_dim + goal_dim  
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return self.max_action * torch.tanh(x)


# Gaussian Goal-conditional Policy
class Gaussian(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_dim = state_dim + goal_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std
    
    def sample(self, state, goal):
        mean, std = self.forward(state, goal)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z)
        # Enforcing action bounds
        log_prob -= torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean
    
    def log_prob(self, state, goal, action):
        eps = 1e-6
        clipped_action = torch.clamp(action / self.max_action, -1 + eps, 1 - eps)
        pre_tanh_value = 0.5 * torch.log((1 + clipped_action) / (1 - clipped_action))

        mean, std = self.forward(state, goal)
        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(pre_tanh_value)
        log_prob -= torch.log(1 - clipped_action.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob
    
    def get_entropy(self, state, goal):
        mean, std = self.forward(state, goal)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy()  # shape: [batch_size, action_dim]
        return entropy.sum(dim=-1, keepdim=True)  # total entropy per sample



