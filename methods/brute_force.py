import numpy as np
import heapq


def least_squares(X, y):
	cov_x = np.matmul(X.T, X)
	cov_xy = np.matmul(X.T, y)
	return np.squeeze(np.dot(np.linalg.inv(cov_x), cov_xy))


def pooled_least_squares(xs, ys):
	return least_squares(np.concatenate(xs, 0), np.concatenate(ys, 0))


def calc_fair_ll_loss(var_set, gamma, covs_xx, covs_xy, pen_coeff=0.0):
	num_envs = len(covs_xx)
	A = np.zeros((len(var_set), len(var_set)))
	b = np.zeros((len(var_set), 1))
	c = 0
	for i in range(num_envs):
		xx = covs_xx[i][var_set, :]
		xx = xx[:, var_set]
		xy = covs_xy[i][var_set, :]
		A = A + (1 + gamma) * xx / num_envs
		b = b + (1 + gamma) * xy / num_envs
		c = c + gamma * np.matmul(np.matmul(np.transpose(xy), np.linalg.inv(xx)), xy) / num_envs

	cur_beta = np.matmul(np.linalg.inv(A), b)
	cur_loss = 0.5 * np.matmul(np.matmul(np.transpose(cur_beta), A), cur_beta) - \
				np.matmul(np.transpose(cur_beta), b) + 0.5 * c + pen_coeff * len(var_set)
	return cur_beta, cur_loss


def calc_eills_loss(var_set, gamma, covs_xx, covs_xy, pen_coeff):
	num_envs = len(covs_xx)
	A = np.zeros((len(var_set), len(var_set)))
	b = np.zeros((len(var_set), 1))
	c = 0
	for i in range(num_envs):
		xx = covs_xx[i][var_set, :]
		xx = xx[:, var_set]
		xy = covs_xy[i][var_set, :]
		A = A + xx / num_envs + gamma * np.matmul(xx, xx) / num_envs
		b = b + xy / num_envs + gamma * np.matmul(xx, xy) / num_envs
		c = c + gamma * np.matmul(np.transpose(xy), xy) / num_envs

	cur_beta = np.matmul(np.linalg.inv(A), b)
	cur_loss = 0.5 * np.matmul(np.matmul(np.transpose(cur_beta), A), cur_beta) - \
				np.matmul(np.transpose(cur_beta), b) + 0.5 * c + pen_coeff * len(var_set)
	return cur_beta, cur_loss


def varset_string_to_numpy(set_str):
	var_set = []
	for i in range(len(set_str)):
		if set_str[i] == '1':
			var_set.append(i)
	return np.array(var_set, dtype=np.int)


def set_to_string(var_set, dim_x):
	myset = set(var_set)
	set_str = ''
	for i in range(dim_x):
		set_str += str(int(i in myset))
	return set_str


def beta_local_to_global(str_set, cand_beta):
	global_beta = np.zeros((len(str_set), 1))
	local_index = 0
	for i in range(len(str_set)):
		if str_set[i] == '1':
			global_beta[i] = cand_beta[local_index]
			local_index += 1
	return global_beta


def support_set(beta):
	var_set = []
	for i in range(np.shape(beta)[0]):
		if np.abs(beta[i]) > 1e-6:
			var_set.append(i)
	return var_set


def greedy_search(x_list, y_list, gamma, iters=1000, loss_type='fair', cand_sets=[], show_log=True):
	num_envs = len(x_list)
	covs_xx = []
	covs_xy = []
	covs_yy = []
	dim_x = np.shape(x_list[0])[1]

	if show_log:
		print(f'Linear model: greedy search, loss type = {loss_type}, d = {dim_x}')
	
	ns = 0
	for i in range(num_envs):
		x, y = x_list[i], y_list[i]
		n = np.shape(x)[0]
		ns += n
		print(f'Env = {i}, number of samples = {n}')
		covs_xx.append(np.matmul(np.transpose(x), x) / n)
		covs_xy.append(np.matmul(np.transpose(x), y) / n)
		covs_yy.append(np.matmul(np.transpose(y), y) / n)

	null_loss = sum([yy for yy in covs_yy]) / num_envs

	min_loss = 0
	min_beta = np.zeros((dim_x, 1))
	min_set = []

	losses, visited = {}, {}

	calc_loss = calc_fair_ll_loss
	if loss_type == 'eills':
		calc_loss = calc_eills_loss

	pooled_lse = pooled_least_squares(x_list, y_list)
	set1 = ''
	for i in range(dim_x):
		set1 += str(int(np.abs(pooled_lse[i]) > 10 * np.log(dim_x)/ns))
	print(set1)
	pen_coeff = 6 * np.log(dim_x) / ns * num_envs

	init_sets = ['0' * dim_x, set1]
	for cand_set in cand_sets:
		init_sets.append(set_to_string(cand_set, dim_x))

	candidates_heap = []
	for a in init_sets:
		_, v = calc_loss(varset_string_to_numpy(a), gamma, covs_xx, covs_xy, pen_coeff)
		candidates_heap.append((float(v), a))
		losses[a] = v
	
	heapq.heapify(candidates_heap)

	for it in range(iters):
		_, cur_set = heapq.heappop(candidates_heap)
		visited[cur_set] = 1
		for j in range(dim_x):
			if cur_set[j] == '0':
				# index j not in current set, consider to incorporate it in
				cand_set = cur_set[:j] + '1' + cur_set[j+1:]
			else:
				# index j in current set, consider to eliminate it out
				cand_set = cur_set[:j] + '0' + cur_set[j+1:]
			
			if cand_set not in losses:
				cand_beta, cand_loss = calc_loss(varset_string_to_numpy(cand_set), gamma, covs_xx, covs_xy, pen_coeff)
				losses[cand_set] = cand_loss
				heapq.heappush(candidates_heap, (cand_loss + 1.0*(cand_loss > losses[cur_set]), cand_set))
				if cand_loss < min_loss:
					min_loss = cand_loss
					min_set = varset_string_to_numpy(cand_set)
					min_beta = beta_local_to_global(cand_set, cand_beta)

	print(f'Greedy search [gamma = {gamma}]: var_set = {min_set}, loss = {min_loss + null_loss}')
	return np.squeeze(min_beta)



def brute_force(x_list, y_list, gamma, loss_type='fair', show_log=False):
	num_envs = len(x_list)
	covs_xx = []
	covs_xy = []
	covs_yy = []
	dim_x = np.shape(x_list[0])[1]

	if show_log:
		print(f'Linear model: brute force search, loss type = {loss_type}, d = {dim_x}')
	
	for i in range(num_envs):
		x, y = x_list[i], y_list[i]
		n = np.shape(x)[0]
		if show_log:
			print(f'Env = {i}, number of samples = {n}')
		covs_xx.append(np.matmul(np.transpose(x), x) / n)
		covs_xy.append(np.matmul(np.transpose(x), y) / n)
		covs_yy.append(np.matmul(np.transpose(y), y) / n)

	null_loss = sum([yy for yy in covs_yy]) / num_envs

	min_loss = 0
	min_beta = np.zeros((dim_x, 1))
	min_set = []

	calc_loss = calc_fair_ll_loss
	if loss_type == 'eills':
		calc_loss = calc_eills_loss

	for sel in range(2 ** dim_x):
		var_set = []
		for j in range(dim_x):
			if (sel & (2 ** j)):
				var_set.append(j)
		
		if len(var_set) == 0:
			cur_loss = 0
			cur_beta = np.zeros((dim_x, 1))
		else:
			cur_beta, cur_loss = calc_loss(np.array(var_set, dtype=np.int32), gamma, covs_xx, covs_xy, 0)

		if cur_loss < min_loss:
			min_loss = cur_loss
			min_set = var_set
			min_beta = np.zeros((dim_x, 1))
			for i, idx in enumerate(var_set):
				min_beta[idx] = cur_beta[i]
	if show_log:
		print(f'Brute force search [gamma = {gamma}]: var_set = {min_set}, loss = {min_loss + null_loss}')
	return np.squeeze(min_beta)


