#to implement the importance sampling algorithm to simulate monte_carlo
import numpy as np
import lin_cong as lc


# Monte Carlo integration with importance sampling
def monte_carlo_importance_sampling(func, imp_sampling_func, lower_bound, upper_bound, num_samples):
    samples = np.array(lc.rand_lin_cong(n = num_samples))
    samples = lower_bound + (upper_bound - lower_bound)*samples
    weights = imp_sampling_func(samples) / (upper_bound - lower_bound)  # Importance sampling weights
    integral_estimate = np.mean(func(samples) / weights)
    return integral_estimate


# print("Monte Carlo integral estimate with importance sampling:", integral_estimate)
