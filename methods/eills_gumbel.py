import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from data.utils import MultiEnvDataset
from methods.modules import *


class EILLSLinearModel(torch.nn.Module):
	def __init__(self, input_dim):
		super(EILLSLinearModel, self).__init__()
		self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
		self.x_mean = torch.tensor(np.zeros((1, input_dim))).float()
		self.x_std = torch.tensor(np.ones((1, input_dim))).float()

	def standardize(self, train_x):
		self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
		self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()

	def forward(self, x):
		x = (x - self.x_mean) / self.x_std
		y = self.linear(x)
		return y


def eills_sgd_gumbel(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=2, niters_g=1, offset=-2,
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, log=False):
	'''
		Implementation of EILLS estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  EILLS Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = EILLSLinearModel(dim_x)
	model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.linear.parameters(), lr=learning_rate)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)

	gate_rec, weight_rec, loss_rec = [], [], []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	tau = init_temp
	for it in it_gen:
		# calculate the temperature
		if (it + 1) % 100 == 0:
			tau = max(final_temp, tau * 0.993)

		optimizer_var.zero_grad()
		optimizer_g.zero_grad()
		xs, ys = dataset.next_batch(batch_size)
		gate = model_var.generate_mask((1, tau))

		outs = [model(gate * x) for x in xs]
		loss_lse = sum([torch.mean(torch.square(ys[e] - outs[e])) for e in range(num_envs)])
		loss_reg = 0
		for e in range(num_envs):
			residual = torch.transpose(ys[e] - outs[e], 0, 1)
			l2n = torch.matmul(residual, xs[e]) / xs[e].shape[0]
			#print(l2n.shape)
			loss_reg += torch.sum(torch.square(l2n) * gate)

		loss = loss_lse + hyper_gamma * loss_reg
		loss.backward()

		optimizer_g.step()
		optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = model.linear.weight.detach().cpu()
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			#print(logits, np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(loss.item())
			if log and it % 5000 == 0:
				print(f'gate = {sigmoid(logits)}, lse loss = {loss_lse.item()}, reg loss = {loss_reg.item()}')


	ret = {'weight': weight_rec[-1] * sigmoid(logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,
			'loss_rec': np.array(loss_rec)}

	return ret

