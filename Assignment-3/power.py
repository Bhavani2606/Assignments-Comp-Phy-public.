#to find the largest eigenvalue of a matrix using power iteration method
import numpy as np

def pow_iter(A, num_iterations=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)  # Random initial guess for the eigenvector

    for _ in range(num_iterations):
        x1 = np.dot(A, x)
        eigenvalue = np.linalg.norm(x1)
        x1 = x1 / eigenvalue  # Normalize the eigenvector estimate
        if np.linalg.norm(x - x1) < tol:
            break
        x = x1

    return eigenvalue, x

