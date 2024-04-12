import numpy as np
import math

def rand_lin_cong(a, c, m, n):#a is the seed, m is the period parameter, n is the number of random numbers required, rng is a 1-d array containing the lower and the upper bound of the random numbers generated.
    x0 = 123
    arr = np.zeros((n))
    arr[0] = x0
    for i in range (1, n):
        arr[i] = ((a*arr[i-1] + c)%m)
    return arr/m

#print(rand_lin_cong(1103515245, 12345, 2**32, 100))
# x = np.array(rand_lin_cong(1103515245, 12345, 2**32, 100))
# y = -(math.pi)/2 + math.pi*x
# print(y)