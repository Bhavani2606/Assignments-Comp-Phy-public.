import numpy as np
import math
import matplotlib.pyplot as plt

def rand_lin_cong(a=1103515245, c=12345, m = 2**32, n=10):#a is the seed, m is the period parameter, n is the number of random numbers required, rng is a 1-d array containing the lower and the upper bound of the random numbers generated.
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

def plot_histogram_and_frequency(data, bins=None):
    plt.figure(figsize=(10, 5))
    counts, edges, _ = plt.hist(data, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.plot(edges[:-1], counts, marker='o', linestyle='-')
    plt.title('Frequency Plot')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
