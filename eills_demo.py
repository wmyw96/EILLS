from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
import numpy as np
from demo_wrapper import *

##############################################
#
#                Batch Tests
#
##############################################


dim_z = dim_x + 1
models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

candidate_n = [100, 300, 700, 1000, 2000]
set_s_star = [0, 1, 2]
set_g = [6, 7, 8]
set_lse = [6, 7, 8, 9]

sets_interested = [
	set_s_star,
	set_g,
	set_lse
]

num_repeats = 300

np.random.seed(0)

methods = [
	eills,
	fair,
	eills_refit,
	lse_s_star,
	lse_gc,
	oracle_icp,
	oracle_irm,
	oracle_anchor,
	causal_dantzig,
	erm
]

result = np.zeros((len(candidate_n), num_repeats, len(methods), dim_x))

for (ni, n) in enumerate(candidate_n):
	for t in range(num_repeats):
		print(f'Running Case: n = {n}, t = {t}')
		# generate data
		xs, ys = [], []
		oracle_var = 0
		for i in range(2):
			x, y, _ = models[i].sample(n)
			xs.append(x)
			ys.append(y)

		for mid, method in enumerate(methods):
			beta = method(xs, ys, true_coeff)
			
			# restore the estimated coeffs
			result[ni, t, mid, :] = beta

np.save('eills_demo.npy', result)


