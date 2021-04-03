import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.matlib

# Sum of Bernoulli = Binomial
p = 0.5
epsilon = 0.01
Nset = np.round(np.logspace(2,5,100)).astype(int)
x = np.zeros((10000,Nset.size))
prob_simulate  = np.zeros(100)
prob_chernoff = np.zeros(100)
prob_hoeffding = np.zeros(100)
for i in range(Nset.size):
  N = Nset[i]
  x[:,i] = stats.binom.rvs(N, p, size=10000)/N
  prob_simulate[i]  = np.mean((np.abs(x[:,i]-p)>epsilon).astype(float))
  prob_chernoff[i] = 2 ** (-N * (1 + (0.5 + epsilon) * np.log2(0.5 + epsilon) + (0.5 - epsilon) * np.log2(0.5 - epsilon)))
  prob_hoeffding[i] = 2*np.exp(-2*N*epsilon**2)

plt.loglog(Nset, prob_simulate,'x')
plt.loglog(Nset, prob_hoeffding, label="Hoeffding")
plt.loglog(Nset, prob_chernoff, label="Chernoff")
plt.legend()
plt.savefig("3b")