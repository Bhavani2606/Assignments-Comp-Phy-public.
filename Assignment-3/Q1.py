import numpy as np
import power as pw
import QR as qr

A = np.array([[4, 2/3, -4/3, 4/3], [2/3, 4, 0, 0], [-4/3, 0, 6, 2], [4/3, 0, 2, 6]])
print("The largest eigenvalue using Power iteration method is \n", round(pw.pow_iter(A)[0], 3))
print("\nThe eigen values using QR factorization (by Gram Schmidt orthogonalization) is\n")
print(qr.eignevlue(A)[0])