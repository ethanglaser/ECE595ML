import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def partBC(end, path):
    train_cat = np.matrix(np.loadtxt('../Data/data/train_cat.txt', delimiter = ','))
    train_grass = np.matrix(np.loadtxt('../Data/data/train_grass.txt', delimiter = ','))
    cov0 = np.cov(train_grass)
    cov1 = np.cov(train_cat)
    mu0 = np.mean(train_grass, axis=1)
    mu1 = np.mean(train_cat, axis=1)
    _, catlen = train_cat.shape
    _, grasslen = train_grass.shape
    pi0 = grasslen / (grasslen + catlen)
    pi1 = catlen / (grasslen + catlen)
    print("i: mu1 and mu0")
    print(mu1[:2], mu0[:2])
    print("ii: sigma1 and sigma0")
    print(cov0[:2,:2], cov1[:2, :2])
    print("iii: pi1 and pi0")
    print(pi1, pi0)
    det0 = np.linalg.det(cov0)
    det1 = np.linalg.det(cov1)
    inv0 = np.linalg.inv(cov0)
    inv1 = np.linalg.inv(cov1)
    const0 = np.log(pi0) - 0.5 * np.log(det0)
    const1 = np.log(pi1) - 0.5 * np.log(det1)

    Y = plt.imread(path) / 255
    if len(Y.shape) > 2:
        Y = rgb2gray(Y)
        plt.imsave('updated.png', Y, cmap='gray')
    M,N = Y.shape
    newY = np.zeros((M-8, N-8))
    for i in range(M-8):
        for j in range(N-8):
            block = np.reshape(Y[i:i+8, j:j+8], (64,1))
            if (-0.5 * (block - mu1).T * inv1 * (block - mu1) + const1) > (-0.5 * (block - mu0).T * inv0 * (block - mu0) + const0):
                newY[i, j] = 1

    plt.imsave(end + '.png', newY, cmap='gray')
    return newY

def partD(newimg):
    r, c = newimg.shape
    truth = plt.imread("../Data/data/truth.png")[:r, :c]
    error = np.sum(np.multiply((truth - newimg), (truth - newimg)).reshape(-1, 1)) / (r * c)
    print("d: MAE")
    print(error)

if __name__ == "__main__":
    newimg = partBC('new', '../Data/data/cat_grass.jpg')
    partD(newimg)
    partBC('test2', '../Data/data/img.jpg')
