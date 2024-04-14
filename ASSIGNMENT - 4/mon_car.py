#to implement monte carlo method using pseudo random number generator.
import lin_cong as lc
import numpy as np
import matplotlib.pyplot as plt
import math 



def monte_carlo(func, a, b, step, lc_a = 1103515245, lc_c = 12345, lc_m = 2**32):
    x = np.array(lc.rand_lin_cong(a = lc_a, c = lc_c, m = lc_m, n = step))
    x = a + (b-a)*x
    my_func = np.vectorize(func)
    y = my_func(x)
    return (b-a)*(np.sum(y))/step


