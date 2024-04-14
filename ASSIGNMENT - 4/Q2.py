import mon_car as mc
import matplotlib.pyplot as plt
import math
import numpy as np

def func(x):
    return math.cos(x)

nos = []
int1 = []
int2 = []
int3 = []
i = 1000
while i <= 100000:
    nos.append(i)
    int1.append(mc.monte_carlo(func=func, a = -(math.pi)/2, b = (math.pi)/2, step = i))
    int2.append(mc.monte_carlo(func=func, a = -(math.pi)/2, b = (math.pi)/2, step = i, lc_a = 65, lc_c = 0, lc_m = 1021))
    int3.append(mc.monte_carlo(func=func, a = -(math.pi)/2, b = (math.pi)/2, step = i, lc_a = 572, lc_c = 0, lc_m = 16381))
    i += 1000

plt.plot(nos, int1)
plt.xlabel("Number of steps")
plt.ylabel("Numerical value")
plt.title("Plot for convergence of integrand with number of steps. The pRNG parameters are a = 1103515245, c = 12345, m = 2^32")
plt.show()

plt.plot(nos, int2)
plt.xlabel("Number of steps")
plt.ylabel("Numerical value")
plt.title("Plot for convergence of integrand with number of steps. a = 65, c = 0, m = 16381")
plt.show()

plt.plot(nos, int3)
plt.xlabel("Number of steps")
plt.ylabel("Numerical value")
plt.title("Plot for convergence of integrand with number of steps. a = 572, m = 16381, c = 0")
plt.show()