import mon_car as mc
import matplotlib.pyplot as plt
import math
import numpy

def func(x):
    return math.cos(x)

nos = []
int = []
i = 1000
while i <= 300000:
    nos.append(i)
    int.append(mc.monte_carlo(func, -(math.pi)/2, (math.pi)/2, i))
    i += 1000

plt.plot(nos, int)
plt.xlabel("Number of steps")
plt.ylabel("Numerical value")
plt.title("Plot for convergence of integrand with number of steps")
plt.show()