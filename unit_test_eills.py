from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
from methods.eills_gumbel import *
import numpy as np
import time
from utils import get_linear_SCM, get_SCM, get_nonlinear_SCM

mode = 1

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16)
rc('text', usetex=True)

if mode == 0:
	candidate_n = [200, 500, 1000, 2000, 5000]

	num_repeats = 50

	np.random.seed(0)

	methods = [
		eills_sgd_gumbel,
		lse_s_star,
		lse_s_rd,
		erm
	]

	result = np.zeros((len(candidate_n), num_repeats, len(methods) + 2, 70))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			#generate random graph with 20 nodes
			models, true_coeff, parent_set, child_set, offspring_set = \
				get_linear_SCM(num_vars=71, num_envs=2, y_index=35, 
								min_child=5, min_parent=5, nonlinear_id=5, 
								bias_greater_than=0.5, same_var=False, log=False)
			
			result[ni, t, 0, :] = true_coeff

			# generate data
			xs, ys = [], []
			for i in range(2):
				x, y, _ = models[i].sample(n)
				xs.append(x)
				ys.append(y)

			for mid, method in enumerate(methods):
				if mid == 0:
					packs = eills_sgd_gumbel(xs, ys, hyper_gamma=10, learning_rate=1e-3, 
												niters=50000, batch_size=64, init_temp=5,
												final_temp=0.1, log=False)
					beta = packs['weight']
					mask = packs['gate_rec'][-1] > 0.8

					# Refit using LS
					full_var = (np.arange(70))
					var_set = full_var[mask].tolist()
					beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, 70)

					result[ni, t, mid + 1, :] = beta
					result[ni, t, len(methods) + 1, :] = beta3
				else:
					beta = method(xs, ys, true_coeff)

					result[ni, t, mid + 1, :] = beta
				
				print(f'method {mid}, l2 error = {np.sum(np.square(true_coeff - beta))}')
			print(f'method {len(methods)}, l2 error = {np.sum(np.square(true_coeff - result[ni, t, len(methods) + 1, :]))}')
			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s')

	np.save('uni_unit_test_0.npy', result)

else:
	results = np.load('uni_unit_test_0.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [200, 500, 1000, 2000, 5000]
	method_name = ["EILLS-GB", "EILLS-RF", "Oracle", r"Semi-Oracle", "Pool-LS"]
	method_idx = [0, 4, 1, 2, 3]

	lines = [
		'solid',
		'solid',
		'dashed',
		'dashed',
		'dashed'
	]

	markers = [
		'D',
		's',
		'^',
		'v',
		'x',
	]

	colors = [
		'#9acdc4',
		'#05348b',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
	]

	fig = plt.figure(figsize=(6, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$", fontsize=22)

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				error = np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :]))
				if error > 0.2 and mid != 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.median(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])
	ax1.set_yscale("log")
	ax1.set_xscale("log")
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	ax1.set_xlabel('$n$', fontsize=22)

	ax1.legend(loc='best')
	plt.show()
