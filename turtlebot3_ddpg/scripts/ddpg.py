#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacity of replay buffer
        self.buffer = []  # Replay buffer
        self.position = 0  # Counting position
    
    def push(self, state, action, reward, next_state, done):
        ''' Replay buffer is a queue 
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # Sample a batch of transitions
        state, action, reward, next_state, done =  zip(*batch)  # Unzip the samples
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' Return the current size 
        '''
        return len(self.buffer)
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3): 
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DDPG:
    def __init__(self, n_states, n_actions, cfg):
        self.device = torch.device(cfg.device)
        self.critic = Critic(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.target_critic = Critic(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)
        # Copy parameters to target networks 
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # Soft update
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]

    def update(self):
        if len(self.memory) < self.batch_size: # Don't update when replay buffer is small
            return
        # Sample experiences from replay buffer 
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # Convert to tensor
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # Calculate losses
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        # Update network parameters
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # Soft update the target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
            
    def save(self,path):
        torch.save(self.actor.state_dict(), path+'actor_checkpoint.pt')
        torch.save(self.critic.state_dict(), path+'critic_checkpoint.pt')

    def load(self,path):
        self.actor.load_state_dict(torch.load(path+'actor_checkpoint.pt')) 
        self.critic.load_state_dict(torch.load(path+'critic_checkpoint.pt')) 
        # self.actor.load_state_dict(torch.load(path+'checkpoint.pt'))
