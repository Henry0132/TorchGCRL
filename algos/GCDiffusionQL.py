import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils import to_torch
from utils.helpers import EMA
from models.model import GCQFunction
from models.diffusion import Diffusion, MLP

class GCDiffusionQL(object):
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
                 alpha=2.5,
                 T=20,
                 beta_schedule='vp',
                 step_start_ema=1000,
                 ema_decay=0.995,
                 update_ema_every=5,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):
        self.device = device

        self.mlp = MLP(state_dim, action_dim, goal_dim, device=device)

        self.actor = Diffusion(state_dim, action_dim, goal_dim, self.mlp, max_action, beta_schedule, T).to(device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.critic = GCQFunction(state_dim, goal_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.discount = discount
        self.tau = tau

        self.alpha = alpha

        self.T = T
        self.beta_schedule=beta_schedule

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, train_dataset, batch_size=256):

        state, action, next_state, reward, value_goal, actor_goal, terminal, mask = to_torch(train_dataset.sample(batch_size), device=self.device)

        not_done = 1. - terminal

        # Q Function Training
        current_Q1, current_Q2 = self.critic.forward(state, value_goal, action)

        with torch.no_grad():
            next_action = self.ema_model(next_state, value_goal)
            target_Q = self.critic_target.q_min(next_state, value_goal, next_action)
            target_Q = reward.unsqueeze(-1) + not_done.unsqueeze(-1) * self.discount * target_Q

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # Policy Training
        bc_loss = self.actor.loss(action, state, actor_goal)
        new_action = self.actor(state, actor_goal)

        q1_new_action, q2_new_action = self.critic(state, new_action, actor_goal)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = bc_loss + self.alpha * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        # Step Target network
        if self.step % self.update_ema_every == 0:
                self.step_ema()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1
            
        return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item()
            }
    
    def sample_action(self, state, goal, device=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        goal_rpt = torch.repeat_interleave(goal, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt, goal_rpt)
            q_value = self.critic_target.q_min(state_rpt, goal_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()







