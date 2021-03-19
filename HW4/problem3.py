import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


def partB():
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    row0, col0 = c0.shape
    row1, col1 = c1.shape
    x = np.column_stack((np.vstack((c0, c1)), np.ones(row0 + row1).reshape(-1, 1)))
    y = np.vstack((np.zeros(row0).reshape(-1, 1), np.ones(row1).reshape(-1, 1)))
    lambd = 0.0001
    theta = cvx.Variable((3,1))
    loss = - cvx.sum(cvx.multiply(y, x @ theta)) + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((row1 + row0,1)), x @ theta]), axis=1 ) )
    reg = cvx.sum_squares(theta)
    prob = cvx.Problem(cvx.Minimize(loss/(row1 + row0) + lambd*reg))
    prob.solve()
    w = theta.value
    print(w)
    return w

def partC(theta):
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    plt.figure()
    plt.scatter(c0[:, 0], c0[:, 1], c='b')
    plt.scatter(c1[:, 0], c1[:, 1], c='r')
    xplot = np.linspace(-1, 7, 3)
    plt.plot(xplot, (theta[2] + xplot * theta[0]) / (-theta[1]))
    plt.savefig("3c")

def partD():
    c0 = np.loadtxt("../Data/data/homework4_class0.txt")
    c1 = np.loadtxt("../Data/data/homework4_class1.txt")
    cov0 = np.cov(c0.T)
    cov1 = np.cov(c1.T)
    mu0 = np.mean(c0, axis=0)
    mu1 = np.mean(c1, axis=0)
    len0, _ = c0.shape
    len1, _ = c1.shape
    pi0 = len0 / (len0 + len1)
    pi1 = len1 / (len0 + len1)
    sqdet0 = np.sqrt(np.linalg.det(cov0))
    sqdet1 = np.sqrt(np.linalg.det(cov1))
    inv0 = np.linalg.inv(cov0)
    inv1 = np.linalg.inv(cov1)
    xset = np.linspace(-5,10,100)
    yset = np.linspace(-5,10,100)
    output = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            block = np.matrix([xset[i], yset[j]])
            output[i, j] = np.log((sqdet0 / sqdet1) * np.exp(0.5 * ((block - mu0) * inv0 * (block - mu0).T - (block - mu1) * inv1 * (block - mu1).T)))
    plt.figure()
    plt.scatter(c0[:,0], c0[:,1],marker='o',s=20)
    plt.scatter(c1[:,0], c1[:,1],marker='+',s=60)
    plt.contour(xset, yset, output>pi0/pi1, linewidths=0.5, colors='k')
    plt.savefig("3d")


if __name__ == "__main__":
    theta = partB()
    partC(theta)
    partD()
