from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
]

dim_x = 12
#n = 100
hyper_gamma = 20
num_simulations = 500

np.random.seed(10)

true_var = {0, 1, 2}
spurious_var = {6, 7, 8}

n = 300

dim_z = dim_x + 1
models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]

true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))
X1, y1, _ = models[0].sample(n)
X2, y2, _ = models[1].sample(n)
cnt_true_local, cnt_spur_local = 0, 0

candidate_gamma = [0, 1, 2, 3, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
				   65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
				   2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

betas = np.zeros((len(candidate_gamma), dim_x))
for i, gamma in enumerate(candidate_gamma):
	betas[i, :] = brute_force([X1, X2], [y1, y2], gamma, loss_type='eills')

#np.save("mybeta.npy", betas)

fig = plt.figure(figsize=(6, 6))
linetypes = ['solid', 'dashed', 'dotted']
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.15, right=0.98)

plt.axvline(15, color='gray', linestyle='dotted')
plt.axvline(3000, color='gray', linestyle='dotted')

for i in range(3):
	plt.plot(candidate_gamma, betas[:, i], color=color_tuple[2])

for i in range(3):
	plt.plot(candidate_gamma, betas[:, i+4], color=color_tuple[1], alpha=(1-i/6*0.8)*0.6, linestyle='dashed')
for i in range(3):
	plt.plot(candidate_gamma, betas[:, i+9], color=color_tuple[1], alpha=(1-(i+3)/6*0.8)*0.6, linestyle='dashed')

for i in range(3):
	plt.plot(candidate_gamma, betas[:, i+6], color=color_tuple[0])


plt.text(100, betas[25, 0], r"$x_1$", fontdict={'color': color_tuple[2]})
plt.text(100, betas[25, 1], r"$x_2$", fontdict={'color': color_tuple[2]})
plt.text(100, betas[25, 2], r"$x_3$", fontdict={'color': color_tuple[2]})
plt.text(1, betas[1, 6], r"$x_7$", fontdict={'color': color_tuple[0]})
plt.text(1, betas[1, 7], r"$x_8$", fontdict={'color': color_tuple[0]})
plt.text(1, betas[1, 8], r"$x_9$", fontdict={'color': color_tuple[0]})
plt.text(1, betas[1, 9], r"$x_{10}$", fontdict={'color': color_tuple[1]})
plt.text(100, -0.1, r"$x_{5}$", fontdict={'color': color_tuple[1]})

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\hat{\beta}_j$")

plt.xscale("log")
plt.savefig("fig3a.pdf")

