import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(-x * x) * 2 #2 * f(x) * f(x) + 1

def s(x):
    return np.exp(-np.abs(x))

def ss(x):
    return np.exp(-(x + 2) * (x + 2)) * 1.5


abs = np.arange(-6, 6, 0.01)
plt.plot(abs,f(abs), label='density f (inliers behavior)', linewidth=4)
plt.plot(abs,s(abs), '--', label='accurate scoring function', linewidth=4)
plt.plot(abs,ss(abs), '--', label='unaccurate scoring function', linewidth=4)

plt.legend(fontsize=18)
plt.show()
