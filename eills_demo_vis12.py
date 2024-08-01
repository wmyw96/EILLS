import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]
results = np.load('eills_demo.npy')
dim_x = 12

env1_model = StructuralCausalModel1(dim_x + 1)
env2_model = StructuralCausalModel2(dim_x + 1)
X1_test, _1, _2 = env1_model.sample(10000)
X2_test, _1, _2 = env2_model.sample(10000)
X_cov = np.matmul(X1_test.T, X1_test) / 20000 + np.matmul(X2_test.T, X2_test) / 20000

num_n = results.shape[0]
num_sml = results.shape[1]

vec_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
method_name = [r"$S^*$", r"$G$"]

lines = [
	'dotted',
	'dashed',
]

markers = [
	'+', 'x'
]

colors = [
	'#05348b',
	'#ae1908',
]

fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
ax1.set_ylabel(r"$\|\bar{\Sigma}^{1/2}(\hat{\beta} - \beta^*)\|_2^2$")

dim_x = 12
true_coeff = np.array([3, 2, -0.5] + [0] * 9)

#true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

print(np.mean(np.abs(results[:, :, 0, 0:3]) > 0, axis=(1,2)) * 3)
print(np.mean(np.abs(results[:, :, 0, 6:9]) > 0, axis=(1,2)) * 3)

ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, 0:3]) > 0, axis=(1,2)) * 3, 
			linestyle=lines[0], marker=markers[0], label=method_name[0], color=colors[0])
ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, 6:9]) > 0, axis=(1,2)) * 3, 
			linestyle=lines[1], marker=markers[1], label=method_name[1], color=colors[1])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
plt.ylim((-0.2, 3.2))
ax1.set_ylabel('number of selected variables')

ax1.legend(loc='best')
plt.savefig("fig3c.pdf")
