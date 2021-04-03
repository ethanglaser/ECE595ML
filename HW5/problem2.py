import random
import matplotlib.pyplot as plt
import numpy as np

def partB():
    experiments = []
    myCoin = random.randint(0,999)
    myCoins = []
    ones = []
    mins = []
    for experiment in range(10000):
        current = [sum([random.randint(0,1) for val in range(10)]) / 10 for valval in range(1000)]
        ones.append(current[0])
        myCoins.append(current[myCoin])
        mins.append(min(current))
        experiments.append(current)
    plt.figure()
    plt.title("Histogram of V1")
    plt.hist(ones, bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1])
    #plt.savefig("ones")
    plt.figure()
    plt.title("Histogram of Vrand")
    plt.hist(myCoins, bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1])
    #plt.savefig("rand")
    plt.figure()
    plt.title("Histogram of Vmin")
    plt.hist(mins, bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1])
    #plt.savefig("mins")
    return ones, myCoins, mins

def partC(v1, vrand, vmin, u1, urand, umin):
    epsilon = np.linspace(0,0.5,101).reshape(-1,1)
    xaxis = np.linspace(0,0.5,101)
    p1 = np.abs(np.array(v1) - u1)
    prand = np.abs(np.array(vrand) - urand)
    pmin = np.abs(np.array(vmin) - umin)
    p1 = np.mean(p1 > epsilon, axis=1)
    prand = np.mean(prand > epsilon, axis=1)
    pmin = np.mean(pmin > epsilon, axis=1)
    yaxis = 2 * np.exp(-20 * xaxis ** 2)

    plt.figure()
    plt.title("V1 with Hoeffding")
    plt.xlabel("Epsilon")
    plt.plot(epsilon, p1, label= "P[|V1 - u1|] > epsilon")
    plt.plot(xaxis, yaxis, label="Hoeffding")
    plt.legend()
    plt.savefig("c1")
    plt.figure()
    plt.title("Vrand with Hoeffding")
    plt.xlabel("Epsilon")
    plt.plot(epsilon, prand, label= "P[|Vrand - urand|] > epsilon")
    plt.plot(xaxis, yaxis, label="Hoeffding")
    plt.legend()
    plt.savefig("crand")
    plt.figure()
    plt.title("Vmin with Hoeffding")
    plt.xlabel("Epsilon")
    plt.plot(epsilon, pmin, label= "P[|Vmin - umin|] > epsilon")
    plt.plot(xaxis, yaxis, label="Hoeffding")
    plt.legend()
    plt.savefig("cmin")


if __name__ == "__main__":
    u1, urand, umin = (0.5, 0.5, 0.5) #will need to fix umin
    v1, vrand, vmin = partB()
    partC(v1, vrand, vmin, u1, urand, umin)