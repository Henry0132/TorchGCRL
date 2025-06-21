import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import to_torch
from models.model import Deterministic, GCQFunction

class GCTD3BC(object):
    def __init__(self, 
                 state_dim, 
                 goal_dim, 
                 action_dim, 
                 max_action, 
                 device, 
                 hidden_dim=256, 
                 lr=3e-4, 
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2, 
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
        ):

        self.device = device

        self.actor = Deterministic(state_dim, goal_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = GCQFunction(state_dim, goal_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.discount = discount
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.alpha = alpha

        self.total_it = 0

    def train(self, train_dataset, batch_size=256):
        
        self.total_it += 1

        state, action, next_state, reward, value_goal, actor_goal, terminal, mask = to_torch(train_dataset.sample(batch_size), device=self.device)

        not_done = 1. - terminal

        with torch.no_grad():
             # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state, value_goal) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q = self.critic_target.q_min(next_state, value_goal, next_action)
            target_Q = reward.unsqueeze(-1) + not_done.unsqueeze(-1) * self.discount * target_Q

        # Q Function Training
        current_Q1, current_Q2 = self.critic.forward(state, value_goal, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed Policy Training
        if self.total_it % self.policy_freq == 0:
            action_pred = self.actor.forward(state, actor_goal)

            Q = self.critic.q_1(state, actor_goal, action_pred)

            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(action_pred, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item()
            }
    
    def sample_action(self, state, goal, device=None):
        state = torch.from_numpy(state).float().to(device)
        goal = torch.from_numpy(goal).float().to(device)
        action = self.actor.forward(state, goal)
        return action.cpu().data.numpy().flatten()