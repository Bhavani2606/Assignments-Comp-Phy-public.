#to implement monte carlo method using pseudo random number generator.
import lin_cong as lc
import numpy as np
import matplotlib.pyplot as plt
import math 

def monte_carlo(func, a, b, step):
    x = np.array(lc.rand_lin_cong(1103515245, 12345, 2**32, step))
    x = a + (b-a)*x
    my_func = np.vectorize(func)
    y = my_func(x)
    return (b-a)*(np.sum(y))/step


