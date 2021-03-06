import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from problem1 import process

def partB(x, y):
    thetahat = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return thetahat

def partC(x, y):
    theta = cp.Variable(3)
    cost = cp.sum_squares(x @ theta - y)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    return theta.value

def partE(x, y):
    N = len(x)
    iters = 50000
    theta = np.zeros(3)
    cost = np.zeros(iters)
    xx = np.dot( np.transpose(x), x)
    for itr in range(iters):
        dJ = np.dot(np.transpose(x), np.dot(x, theta)-y)
        dd = dJ
        alpha = np.dot(dJ, dd) / np.dot(np.dot(xx, dd), dd)
        theta = theta - alpha*dd
        cost[itr] = np.linalg.norm(np.dot(x, theta)-y)**2/N
    return theta, cost

def partFH(cost, char):
    x = np.linspace(1,50000,50000)
    plt.figure()
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Cost" + (" Momentum" if char == "h" else ""))
    plt.semilogx(x, cost, linewidth=8)
    plt.savefig("prob2imgs/q5")

def partG(x, y):
    N = len(x)
    iters = 50000
    beta = 0.9
    theta = np.ones(3)
    dJ1 = theta
    cost = np.zeros(iters)
    xx = np.dot( np.transpose(x), x)
    for itr in range(iters):
        dJ = np.dot(np.transpose(x), np.dot(x, theta)-y)
        dd = (1 - beta) * dJ1 + beta * dJ
        alpha = np.dot(dJ, dd) / np.dot(np.dot(xx, dd), dd)
        theta = theta - alpha*dd
        cost[itr] = np.linalg.norm(np.dot(x, theta)-y)**2/N
        dJ1 = dJ
    return theta, cost


if __name__ == "__main__":
    male = process("male_train_data.csv").to_numpy()
    female = process("female_train_data.csv").to_numpy()
    y = np.hstack((np.ones(len(male)), -1 * np.ones(len(female))))
    x = np.hstack((np.ones((len(y), 1)), np.vstack((male, female))))
    B = partB(x, y)
    print(B)
    C = partC(x, y)
    print(C)
    E, cost = partE(x, y)
    print(E)
    partFH(cost, "f")
    G, cost2 = partG(x, y)
    print(G)
    partFH(cost2, "h")
