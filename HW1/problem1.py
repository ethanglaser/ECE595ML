import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

#A.
mu = 0
sigma = 1
x = np.linspace(-3,3,num=50)
y = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
plt.plot(x,y)
plt.savefig("prob1imgs/1a")

#B.
n = 1000
vals = sorted(np.random.normal(0,1,n))
plt.figure()
plt.hist(vals,4)
plt.savefig("prob1imgs/1bii4")
plt.figure()
plt.hist(vals,1000)
plt.savefig("prob1imgs/1bii1000")
mean, std = norm.fit(vals)
plt.figure()
plt.hist(vals,bins=4,density=True)
plt.plot(vals, norm.pdf(vals))
plt.savefig("prob1imgs/1biv4")
plt.figure()
plt.hist(vals,bins=1000,density=True)
plt.plot(vals, norm.pdf(vals))
plt.savefig("prob1imgs/1biv1000")
print(mean, std)

#C.
n = 1000
vals = sorted(np.random.normal(0,1,n))
m = np.linspace(1,200,200)
h = (max(vals) - min(vals)) / m
J = np.zeros(200)
for i in range(200):
    hist, edges = np.histogram(vals, int(m[i]))
    J[i] = (2 / (h[i] * (n - 1))) - (n + 1) / (h[i] * (n - 1)) * np.sum((hist / n) ** 2)
plt.figure()
plt.plot(m, J)
bestbins = np.argmin(J) + 1
plt.savefig("prob1imgs/1ci")
print(bestbins)
plt.figure()
plt.hist(vals,bins=bestbins,density=True)
plt.plot(vals, norm.pdf(vals))
plt.savefig("prob1imgs/1ciii")
