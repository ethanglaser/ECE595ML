import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-5, 5, 6)
y = 1 / (1 + np.exp(-2 * (x + 1)))
plt.plot(x,y,'k')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("my plot")
#plt.show()
plt.savefig('fig')

