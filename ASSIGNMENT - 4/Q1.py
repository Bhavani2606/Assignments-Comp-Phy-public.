import lin_cong as lc
import numpy as np
import matplotlib.pyplot as plt

x = np.array(lc.rand_lin_cong(a = 65, c = 0, m = 1021, n = 2000))
y = np.array(lc.rand_lin_cong(a = 572, m = 16381, c = 0, n = 2000))

lc.plot_histogram_and_frequency(x, 20)
lc.plot_histogram_and_frequency(y, 20)

'''
Output are two frequency polygons and histogram plotsfor random numbers generated between 0 and 1
'''