import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x11 = F.relu(self.l2(x1))
		x1 = self.l3(x11)

		x2 = F.relu(self.l4(xu))
		x22 = F.relu(self.l5(x2))
		x2 = self.l6(x22)
		return x1, x2, x11, x22 #, self.cos(x11, x22)


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 

	def Q2(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l4(xu))
		x1 = F.relu(self.l5(x1))
		x1 = self.l6(x1)
		return x1 

class TD3(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Select action according to policy and add clipped noise 
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2, vec_q1, vec_q2 = self.critic_target(next_state, next_action)
			approx_Q1, approx_Q2, avec_q1, avec_q2 = self.critic(next_state, next_action)
			coeff = self.critic.cos(avec_q1, avec_q2).view(batch_size, -1)
			target_Q = torch.min(target_Q2, target_Q1- coeff*( approx_Q2 - target_Q2) )
			# target_Q =  torch.min( target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2, vec_q1, vec_q2 = self.critic(state, action)
			coeff = self.critic.cos(vec_q1, vec_q2).view(batch_size, -1)

			# diff = current_Q1-current_Q2- (done * discount *(target_Q2-target_Q1)).detach()
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #+ 2*F.mse_loss(diff, torch.zeros([batch_size, 1], dtype=torch.float32, device='cuda'))
			# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, torch.zeros([batch_size, 1], dtype=torch.float32, device='cuda')) 
			# critic_loss = F.mse_loss(current_Q1, target_Q) + 4*F.mse_loss(diff, torch.zeros([batch_size, 1], dtype=torch.float32, device='cuda'))
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			
			# Delayed policy updates
			if it % policy_freq == 0:
				a = self.actor(state)
				approx_Q1, approx_Q2, vec_q1, vec_q2 = self.critic(state, a)
				coeff = self.critic.cos(vec_q1, vec_q2).view(batch_size, -1)
				exp_q2 = self.critic_target.Q2(state, a)
				# Compute actor loss
				# actor_loss = -torch.max(self.critic.Q2(state, a), self.critic.Q1(state, a)).mean()
				actor_loss = -( self.critic.Q1(state, a)- (coeff*(approx_Q2-exp_q2)).detach() ).mean()
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				# add clip here
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
