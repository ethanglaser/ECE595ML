import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx



def partB():
    train_cat = np.matrix(np.loadtxt('../Data/data/train_cat.txt', delimiter = ','))
    train_grass = np.matrix(np.loadtxt('../Data/data/train_grass.txt', delimiter = ','))
    cov0 = np.cov(train_grass)
    cov1 = np.cov(train_cat)
    cov1 = cov0
    mu0 = np.mean(train_grass, axis=1)
    mu1 = np.mean(train_cat, axis=1)
    _, catlen = train_cat.shape
    _, grasslen = train_grass.shape
    pi0 = grasslen / (grasslen + catlen)
    pi1 = catlen / (grasslen + catlen)
    #print(cov0[:2,:2], cov1[:2, :2], mu0[:2], mu1[:2], pi0, pi1)
    sqdet0 = np.sqrt(np.linalg.det(cov0))
    sqdet1 = np.sqrt(np.linalg.det(cov1))
    inv0 = np.linalg.inv(cov0)
    inv1 = np.linalg.inv(cov1)


    Y = plt.imread('../Data/data/cat_grass.jpg') / 255
    M,N = Y.shape
    newY = np.zeros((M-8, N-8))
    truth = plt.imread("../Data/data/truth.png")[:M-8, :N-8].reshape(-1,1)
    pos = np.sum(truth)
    neg = (M - 8) * (N - 8) - pos
    pfList = []
    pdList = []
    for i in range(M-8):
        for j in range(N-8):
            block = np.reshape(Y[i:i+8, j:j+8], (64,1))
            newY[i, j] = np.log((sqdet0 / sqdet1) * np.exp(0.5 * ((block - mu0).T * inv0 * (block - mu0) - (block - mu1).T * inv1 * (block - mu1))))
    #print(np.max(newY, 1))
    maxx = 100
    for val in range(-200, 100):
        newimg = newY > val
        #plt.imsave("testimgs/img" + str(val) + '.png', newimg, cmap='gray')
        newimg = newimg.reshape(-1, 1)
        truepos = np.sum((truth + newimg) == 2)
        falsepos = np.sum((newimg - truth) == 1)
        pD = truepos / pos
        pF = falsepos / neg
        pfList.append(pF)
        pdList.append(pD)
    plt.figure()
    plt.plot(pfList, pdList)
    plt.savefig('ROCtest.png')
    newtau = pi0/pi1
    newimg = newY > newtau
    #plt.imsave("testimgs/img" + str(val) + '.png', newimg, cmap='gray')
    newimg = newimg.reshape(-1, 1)
    truepos = np.sum((truth + newimg) == 2)
    falsepos = np.sum((newimg - truth) == 1)
    pdd = truepos / pos
    pff = falsepos / neg
    plt.scatter([pff], [pdd], c='r', linewidths=10)
    plt.savefig('ROCc.png')
    plt.imsave('testtest.png', newY, cmap='gray')
    return newY

def partD():
    train_cat = np.matrix(np.loadtxt('../Data/data/train_cat.txt', delimiter = ',')).T
    train_grass = np.matrix(np.loadtxt('../Data/data/train_grass.txt', delimiter = ',')).T
    catlen, _ = train_cat.shape
    grasslen, _ = train_grass.shape    
    A = np.vstack((train_cat, train_grass))
    B = np.hstack((np.ones(catlen), -1 * np.ones(grasslen))).T
    print(A.shape, B.shape)
    t       = cvx.Variable(64)
    objective   = cvx.Minimize( cvx.sum_squares(A @ t-B) )
    prob        = cvx.Problem(objective)
    prob.solve()
    theta = np.matrix(t.value)
    Y = plt.imread('../Data/data/cat_grass.jpg') / 255
    M, N = Y.shape
    newY = np.zeros((M-8, N-8))
    truth = plt.imread("../Data/data/truth.png")[:M-8, :N-8].reshape(-1,1)
    pos = np.sum(truth)
    neg = (M - 8) * (N - 8) - pos
    pfList = []
    pdList = []
    for i in range(M-8):
        for j in range(N-8):
            block = np.reshape(Y[i:i+8, j:j+8], (64,1))
            newY[i,j] = np.sum(np.multiply(block, theta.T))
    newY = newY.reshape(-1,1)

    for val in range(0,1000):
        tau = -0.001 * val
        newimg = newY > tau
        truepos = np.sum((truth + newimg) == 2)
        falsepos = np.sum((newimg - truth) == 1)
        pD = truepos / pos
        pF = falsepos / neg
        pfList.append(pF)
        pdList.append(pD)
        #print(pD, pF)
    plt.figure()
    plt.plot(pfList, pdList)
    plt.savefig('ROCd.png')
    

if __name__ == "__main__":
    partB()
    #partD()