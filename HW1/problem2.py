import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power

#a
x1 = np.linspace(-1, 5, 100)
x2 = np.linspace(0, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
mu = np.array([[2, 6]])
x = np.array([[X1, X2]])
sigma = np.array([[2,1],[1,2]])
f = np.zeros((len(x1),len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        xtmp = np.array([[x1[i], x2[j]]])
        f[i,j] = 1 / np.sqrt(2 * np.pi * np.linalg.det(sigma)) * np.exp(-0.5 * np.dot(np.dot((xtmp - mu), np.linalg.inv(sigma)), (xtmp - mu).T))
plt.contour(X1, X2, f, colors='black')
plt.savefig("prob2imgs/2aii")

#c
x2 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],5000)
sigma2 = np.array([[2,1],[1,2]])
mu2 = np.array([2,6])
sigma3 = fractional_matrix_power(sigma, 0.5)
print(sigma3)
y2 = np.dot(sigma3, x2.T) + matlib.repmat(mu,5000,1).T
plt.figure()
plt.scatter(x2[:,0], x2[:,1])
plt.savefig("prob2imgs/2ci")
plt.figure()
plt.scatter(y2[0,:],y2[1,:])
plt.savefig("prob2imgs/2cii")
