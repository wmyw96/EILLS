import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper import *
from methods.tools import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]
results = np.load('eills_demo_large.npy')
dim_x = 12

env1_model = StructuralCausalModel1(dim_x + 1)
env2_model = StructuralCausalModel2(dim_x + 1)
X1_test, y1_test, _ = env1_model.sample(1000000)
X2_test, y2_test, _ = env2_model.sample(1000000)
X_cov = np.matmul(X1_test.T, X1_test) / 2000000 + np.matmul(X2_test.T, X2_test) / 2000000
beta1 = least_squares(X1_test, y1_test)
beta2 = least_squares(X2_test, y2_test)
beta0 = np.squeeze(pooled_least_squares([X1_test, X2_test], [y1_test, y2_test]))

num_n = results.shape[0]
num_sml = results.shape[1]

n = 700
#method_name = ['EILLS', "ICP", "Anchor", "IRM", "PLS", r"EILLS+refit"]
#method_idx = [0, 5, 7, 6, 8, 2]
method_name = ['EILLS', "EILLS+refit"]
method_idx = [0, 2]

markers = [
	'D',
	'o',
	'P',
	's',
	'x',
	'<',
]

colors = [
	'#05348b',
	'#ae1908',
	'#9acdc4',
	'#ec813b',
	'#e5a84b',
	'#6bb392'
]

fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.15, left=0.15, right=0.98)

dim_x = 12
true_coeff = np.array([3, 2, -0.5] + [0] * 9)
true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

n_points = 100
betas_e = results[4, :n_points, 0, :]
betas_r = results[4, :n_points, 2, :]

xe = np.sqrt(np.sum(np.square(betas_e[:, [0, 1, 2]]), axis=1)) / np.sqrt(np.sum(np.square(true_coeff)))
ye = np.sqrt(np.sum(np.square(betas_e[:, [6, 7, 8]]), axis=1)) / np.sqrt(np.sum(np.square(beta0[[6, 7, 8, 9]])))
xr = np.sqrt(np.sum(np.square(betas_r[:, [0, 1, 2]]), axis=1)) / np.sqrt(np.sum(np.square(true_coeff)))
yr = np.sqrt(np.sum(np.square(betas_r[:, [6, 7, 8]]), axis=1)) / np.sqrt(np.sum(np.square(beta0[[6, 7, 8, 9]])))

rest_e = np.sum(np.square(betas_e[:, 3:]), axis=1)
rest_r = np.sum(np.square(betas_r[:, 3:]), axis=1)

print(f'eills = {np.mean(rest_e)} +/- {np.std(rest_e)}')
print(f'refit = {np.mean(rest_r)} +/- {np.std(rest_r)}')

ax1.scatter(xe, ye, marker='D', label='EILLS', color=color_tuple[2])
ax1.scatter(xr, yr, marker='o', label='w/ Refit', color=color_tuple[3])

for i in range(n_points):
    plt.arrow(xe[i], ye[i], xr[i] - xe[i], yr[i] - ye[i], width=0.0002, head_width=0.001, head_length=0.001, fc='black', ec='black')


#plt.xlim(left=0)
#plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlim(left=-0.05)
ax1.text(1, 0, r'$\boldsymbol{\beta}^*$', color=color_tuple[0])
ax1.text(np.sqrt(np.sum(np.square(beta0[[0, 1, 2]]))) / np.sqrt(np.sum(np.square(true_coeff))), 
		1, r'$\bar{\boldsymbol{\beta}}$', color=color_tuple[0])
ax1.set_xlabel(r'$\|\hat{\beta}_{S^*}\|_2/\|\beta_{S^*}^*\|_2$')
ax1.set_ylabel(r"$\|\hat{\beta}_{G}\|_2/\|\bar{\beta}_{G}\|_2$")

ax1.legend(loc='best', fontsize=15)
plt.show()
#plt.savefig("l2error_n_sigma.pdf")
