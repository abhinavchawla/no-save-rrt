import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import integrate
import math
mu = 0
variance = 0.1
def normal_distribution_function(x):
    value = -np.inf
    for i in range(0,10,3):
        value = max(value, stats.norm.pdf(x,i*0.1,variance))
    return value

sigma = math.sqrt(variance)
l=0
r=1
x = np.linspace(l, r, 1000)
m = time.time()
res = integrate.quad(normal_distribution_function, l, r)
print(time.time()-m)
print("RESULT",res)
y = []
for i in x:
    y.append(normal_distribution_function(i))
m = time.time()
tmp = []
for i in range(0,1000):
    tmp.append(stats.norm(i, variance))
print(time.time()-m)
m = time.time()
for i in range(0,1000):
    t = tmp[i].pdf(i)
print(time.time()-m)
m = time.time()
for i in range(0,1000):
    t = i*i
print(time.time()-m)
plt.plot(x, y)
plt.show()