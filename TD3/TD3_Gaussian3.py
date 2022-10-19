import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.mean = nn.Linear(300, action_dim)
		self.log_std = nn.Linear(300, action_dim)
		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		# x = self.max_action * torch.tanh(self.l3(x)) 
		mean = self.mean(x)
		# Clamped for numerical stability 
		log_std = self.log_std(x).clamp(-4, 15)
		std = torch.exp(log_std)
		# z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
		return mean, std

	def sample(self, state, epsilon=1e-6):
		mean, std = self.forward(state)
		normal = Normal(mean, std)
		z = normal.rsample()
		action = self.max_action * torch.tanh(z)
		log_pi = normal.log_prob(z) - torch.log(1-action.pow(2) + epsilon)
		log_pi = log_pi.sum(1, keepdim=True)

		return action, log_pi

	def sample2(self, state, epsilon=1e-6):
		mean, std = self.forward(state)
		normal = Normal(mean, std)
		z = normal.sample()
		action = self.max_action * torch.tanh(z)
		log_pi = normal.log_prob(z) - torch.log(1-action.pow(2) + epsilon)
		log_pi = log_pi.sum(1, keepdim=True)

		return action, log_pi

	def likelihood(self, state, action, epsilon=1e-6):
		mean, std = self.forward(state)
		normal = Normal(mean, std)
		# z = normal.sample()
		# action = self.max_action * torch.tanh(z)
		log_pi = normal.log_prob(action)  #- torch.log(1-action.pow(2) + epsilon)
		log_pi = log_pi.sum(1, keepdim=True)

		return log_pi

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


	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 

	def Q2(self, x, u):
		xu = torch.cat([x, u], 1)
		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x2

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


	def select_action(self, state, epsilon=1e-6):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		# return self.actor(state).cpu().data.numpy().flatten()
		with torch.no_grad():
			mean, scale = self.actor(state)
		dist = Normal(mean, scale)
		# action = dist.sample().cpu().data.numpy().flatten()
		# action_log_prob = dist.log_prob(action).cpu().data.numpy().flatten()
		z = dist.sample()
		action = self.max_action*torch.tanh(z)
		log_pi = dist.log_prob(z) #- torch.log(1-action.pow(2) + epsilon)
		# log_pi = log_pi.sum(1, keepdim=True)
		action_log_prob = log_pi.cpu().data.numpy().flatten()
		return action.cpu().detach().squeeze(0).numpy(), action_log_prob.sum()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, gama=0.004):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, upi, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)
			u_pi = torch.FloatTensor(upi).to(device)
			u_pi_target = self.actor.likelihood(state, action)

			# Select action according to policy and add clipped noise 
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			# next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
			next_action, next_log_pi = self.actor_target.sample(next_state)
			next_action = next_action.clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			'''
			vs = 0.0
			nsamples = 5
			for i in range(nsamples):
				action, log_pi = self.actor_target.sample2(state)
				vs =vs + self.critic.Q1(state, action)
			vs = vs / nsamples 
			'''
			learning_rate=0.001
			ratios = torch.exp(u_pi_target.detach()-u_pi.detach())
			# ratios = torch.clamp(ratios,1.0, 1.3)
			# Delayed policy updatesxcv
			if it % policy_freq == 0:

				# Compute actor loss
				action, log_pi = self.actor.sample(state)
				# actor_loss = - (torch.clamp(torch.exp(log_pi)/(torch.exp(u_pi)+1e-8), max=1) * self.critic.Q1(state, action)).mean()
				# advantage = self.critic.Q1(state, action) #- vs
				q1 = self.critic.Q1(state, action)
				#temp = advantage.clone().detach()
				# weight = 1-torch.sigmoid(temp)
				# actor_loss = -(ratios*advantage).mean() # - gama*weight*torch.exp(log_pi)*log_pi).mean()
				actor_loss = -(q1).mean() 
				# Optimize the actor 
				# self.actor_optimizer.zero_grad()
				# actor_loss.backward()
				grad1 = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)
				# add clip here
				'''
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
				# self.actor_optimizer.step()
				grad1 = {}
				for param in self.actor.parameters():
				    grad1[param] = param.grad
				'''
				q2 = self.critic.Q2(state, action)
				actor_loss = -(q2).mean() 
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				with torch.no_grad():
					for idx, param in enumerate(self.actor.parameters()):
						param -= 0.5*learning_rate * (param.grad + grad1[idx])
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
