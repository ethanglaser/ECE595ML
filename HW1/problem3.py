import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import linprog
from matplotlib import pyplot as plt

#a.
N = 50
x = np.linspace(-1,1,N)
a = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
epsilon = np.random.normal(0, 0.2**2, N)
y = a[0]*eval_legendre(0,x) + a[1]*eval_legendre(1,x) + \
  a[2]*eval_legendre(2,x) + a[3]*eval_legendre(3,x) + \
  a[4]*eval_legendre(4,x) + epsilon
plt.figure()
plt.scatter(x,y)
plt.savefig("prob3imgs/3a")

# need part b

#c
X = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), \
                     eval_legendre(2,x), eval_legendre(3,x), \
                     eval_legendre(4,x)))
theta = np.linalg.lstsq(X, y, rcond=None)[0]
t     = np.linspace(-1, 1, 200)
yhat  = theta[0]*eval_legendre(0,t) + theta[1]*eval_legendre(1,t) + \
        theta[2]*eval_legendre(2,t) + theta[3]*eval_legendre(3,t) + \
        theta[4]*eval_legendre(4,t)
plt.scatter(x, y)
plt.plot(t, yhat, 'r')
plt.savefig('prob3imgs/3c')

#d
idx = [10,16,23,37,45]
y[idx] = 5
theta2 = np.linalg.lstsq(X, y, rcond=None)[0]
yhat2  = theta2[0]*eval_legendre(0,t) + theta2[1]*eval_legendre(1,t) + \
        theta2[2]*eval_legendre(2,t) + theta2[3]*eval_legendre(3,t) + \
        theta2[4]*eval_legendre(4,t)
plt.figure()
plt.scatter(x, y)
plt.plot(t, yhat2, 'r')
plt.savefig('prob3imgs/3d')
#line no longer matches the trend of the vast majority of the data, greatly influenced by outliers

#e
X_ = np.column_stack((np.ones(N), x, x**2, x**3, x**4))
A_ = np.vstack((np.hstack((X_,-np.identity(N))),np.hstack((-X_,-np.identity(N)))))
b = np.vstack((y.reshape(-1,1),-y.reshape(-1,1)))
c = np.column_stack((np.zeros((1,5)), np.ones((1,N))))

sol = linprog(c, A_ub=A_, b_ub=b, bounds=(None, None))
beta = sol['x'][:5]
u = sol['x'][5:]
yhat3  = beta[0] + beta[1]*t + beta[2]*t**2 + beta[3]*t**3 + beta[4]*t**4
plt.figure()
plt.scatter(x, y)
plt.plot(t, yhat3, 'r')
plt.savefig('prob3imgs/3f')