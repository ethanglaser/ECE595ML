import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
from problem1 import process


def four(x, y):
    lambd = np.arange(0.1, 10, 0.1)
    thetas = np.zeros(len(lambd))
    costs = np.zeros(len(lambd))
    for index in range(len(lambd)):
        theta = cp.Variable(3)
        objective = cp.Minimize( cp.sum_squares(x @ theta-y) + lambd[index]*cp.sum_squares(theta) )
        prob = cp.Problem(objective)
        prob.solve()
        t = theta.value
        c = np.dot(x, t.reshape(-1, 1)) - y.reshape(-1, 1)
        thetas[index] = np.dot(t.T, t)
        costs[index] = np.dot(c.T, c)
        print(np.sum(x))
    plt.title("Base cost vs. regularization term cost")
    plt.xlabel("L2 Norm of Coefficients")
    plt.ylabel("L2 Norm of Cost")
    plt.plot(thetas, costs)
    plt.savefig("prob4imgs/4a1")
    plt.figure()
    plt.title("Base cost vs. lambda")
    plt.xlabel("Lambda value")
    plt.ylabel("L2 Norm of Cost")
    plt.plot(lambd, costs)
    plt.savefig("prob4imgs/4a2")
    plt.figure()
    plt.title("Regularization term cost vs. lambda")
    plt.ylabel("L2 Norm of Coefficients")
    plt.xlabel("Lambda")
    plt.plot(lambd, thetas)
    plt.savefig("prob4imgs/4a3")


if __name__ == "__main__":
    male = process("male_train_data.csv").to_numpy()
    female = process("female_train_data.csv").to_numpy()
    y = np.hstack((np.ones(len(male)), -1 * np.ones(len(female))))
    x = np.hstack((np.ones((len(y), 1)), np.vstack((male, female))))
    four(x, y)

