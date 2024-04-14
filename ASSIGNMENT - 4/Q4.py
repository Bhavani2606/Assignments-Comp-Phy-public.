import numpy as np
import math
import matplotlib.pyplot as plt
import impo_sampling as i_s

def func(x):
    return math.exp(-2*x)/(1+x**2)

def imp_samp1(x):
    return 0.5

def imp_samp2(x):
    return math.exp(-x)

def imp_samp3(x):
    return math.exp(-x/2)/(2*(1- math.exp(-0.5)))

print(i_s.monte_carlo_importance_sampling(np.vectorize(func), np.vectorize(imp_samp1), 0, 2, 10000))
print(i_s.monte_carlo_importance_sampling(np.vectorize(func), np.vectorize(imp_samp2), 0, 2, 10000))
print(i_s.monte_carlo_importance_sampling(np.vectorize(func), np.vectorize(imp_samp3), 0, 2, 10000))