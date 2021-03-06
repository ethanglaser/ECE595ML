import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from problem2 import partB
from problem1 import process

def threeA(male, female, theta):
    plt.figure()
    plt.scatter(male[:, 0], male[:, 1], c='b', s=0.5)
    plt.scatter(female[:, 0], female[:, 1], c='r', s=0.5)
    xplot = np.linspace(0, 10, 101)
    plt.plot(xplot, (theta[0] + xplot * theta[1]) / (-theta[2]))
    plt.savefig("prob3imgs/3a")

def threeB(male, female, theta):
    fp = np.sum(np.dot(np.column_stack((np.ones((len(female), 1)), female)), theta.reshape(-1,1)) > 0)
    fn = np.sum(np.dot(np.column_stack((np.ones((len(male), 1)), male)), theta.reshape(-1,1)) < 0)
    type1error = 100 * fp / len(female)
    type2error = 100 * fn / len(male)
    tp = len(male) - fn
    print(type1error)
    print(type2error)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(precision, recall)


if __name__ == "__main__":
    male = process("male_train_data.csv").to_numpy()
    female = process("female_train_data.csv").to_numpy()
    y = np.hstack((np.ones(len(male)), -1 * np.ones(len(female))))
    x = np.hstack((np.ones((len(y), 1)), np.vstack((male, female))))
    theta = partB(x, y)
    male = process("male_test_data.csv").to_numpy()
    female = process("female_test_data.csv").to_numpy()
    y = np.hstack((np.ones(len(male)), -1 * np.ones(len(female))))
    x = np.hstack((np.ones((len(y), 1)), np.vstack((male, female))))
    threeA(male, female, theta)
    threeB(male, female, theta)

