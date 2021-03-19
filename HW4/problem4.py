import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from numpy.matlib import repmat


def partA(h):
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    x = np.vstack((c0, c1))
    row0, col0 = c0.shape
    row1, col1 = c1.shape
    kernel = np.zeros((row0 + row1, row0 + row1))
    for i in range(row0 + row1):
        for j in range(row0 + row1):
            kernel[i, j] = np.exp(-np.linalg.norm(x[i] - x[j]) ** 2 / h)
    print(kernel[47:52, 47:52])
    return kernel

def partC(kernel):
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    x = np.vstack((c0, c1))
    row0, col0 = c0.shape
    row1, col1 = c1.shape
    y = np.vstack((np.zeros(row0).reshape(-1, 1), np.ones(row1).reshape(-1, 1))).reshape(-1, 1)
    lambd = 0.001
    alpha = cvx.Variable((row0 + row1, 1))
    loss = - y.T @ kernel @ alpha + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((row1 + row0,1)), kernel @ alpha]), axis=1 ))
    reg = cvx.quad_form(alpha, kernel)
    prob = cvx.Problem(cvx.Minimize(loss/(row1 + row0) + lambd * reg))
    prob.solve()
    w = alpha.value
    print(w[:2])
    return w

def partD(alpha, h):
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    row0, col0 = c0.shape
    row1, col1 = c1.shape
    x = np.column_stack((np.vstack((c0, c1)), np.ones(row0 + row1).reshape(-1, 1)))
    xset = np.linspace(-5,10,100)
    yset = np.linspace(-5,10,100)
    output = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            data = repmat( np.array([xset[j], yset[i], 1]).reshape((1,3)), row0+row1, 1)
            phi  = np.exp( -np.sum( (x-data)**2, axis=1 )/h )
            output[i,j] = np.dot(phi.T, alpha)

    plt.scatter(c0[:,0], c0[:,1],marker='o',s=20)
    plt.scatter(c1[:,0], c1[:,1],marker='+',s=60)
    plt.contour(xset, yset, output>0.5, linewidths=2, colors='k')
    plt.savefig("4d")

if __name__ == "__main__":
    h = 1
    kernel = partA(h)
    alpha = partC(kernel)
    partD(alpha, h)