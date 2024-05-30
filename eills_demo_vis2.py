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
method_name = ['EILLS', "ICP", "Anchor", "IRM", "PLS"]
method_idx = [0, 5, 7, 6, 9]

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
	'#6bb392',
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

y_max = 1.0
for (j, mid) in enumerate(method_idx):
	betas = results[3, :60, mid, :]
	xs = np.sqrt(np.sum(np.square(betas[:, [0, 1, 2]]), axis=1)) / np.sqrt(np.sum(np.square(true_coeff)))
	ys = np.sqrt(np.sum(np.square(betas[:, [6, 7, 8]]), axis=1)) / np.sqrt(np.sum(np.square(beta0[[6, 7, 8]])))
	ax1.scatter(xs, ys, marker=markers[j], label=method_name[j], color=colors[j])
	y_max = max(y_max, np.max(ys))

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlim(left=-0.05)
#ax1.set_ylim(top=y_max + 0.1)
ax1.text(1, 0, r'$\boldsymbol{\beta}^*$', color=color_tuple[0])
ax1.text(np.sqrt(np.sum(np.square(beta0[[0, 1, 2]]))) / np.sqrt(np.sum(np.square(true_coeff))), 
		1, r'$\bar{\boldsymbol{\beta}}$', color=color_tuple[0])
ax1.set_xlabel(r'$\|\hat{\beta}_{S^*}\|_2/\|\beta_{S^*}^*\|_2$')
ax1.set_ylabel(r"$\|\hat{\beta}_{G}\|_2/\|\bar{\beta}_{G}\|_2$")


ax1.legend(loc='best')
plt.show()
#plt.savefig("l2error_n_sigma.pdf")
