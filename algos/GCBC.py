import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Deterministic
from utils.utils import to_torch


class GCBC(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, device, lr=3e-4, hidden_dim=256):

        self.device = device

        self.actor = Deterministic(state_dim, goal_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def train(self, train_dataset, batch_size=256):
        state, action, _, _, _, actor_goal, _, _ = to_torch(train_dataset.sample(batch_size), device=self.device)
        
        action_pred = self.actor.forward(state, actor_goal)
        actor_loss = F.mse_loss(action_pred, action)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item()
        }
    
    def sample_action(self, state, goal, device=None):
        state = torch.from_numpy(state).float().to(device)
        goal = torch.from_numpy(goal).float().to(device)
        action = self.actor.forward(state, goal)
        return action.cpu().data.numpy().flatten()
