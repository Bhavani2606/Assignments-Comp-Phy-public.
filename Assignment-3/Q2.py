import numpy as np
import matplotlib.pyplot as plt
import gauss_siedel as gs


ylist = np.array([0.486, 0.866, 0.944, 1.144, 1.103, 1.202, 1.166, 1.191, 1.124, 1.095, 1.122, 1.102, 1.099, 1.017, 1.111, 1.117, 1.152, 1.265, 1.380, 1.575, 1.875])
xlist = []
i =0
while i <= 1.05:
    xlist.append(i)
    i += 0.05
xlist = np.array(xlist)

def pol_basis(x):
    return np.array([np.ones_like(x), x**1, x**2, x**3])

def cheby_basis(x):
    return np.array([np.ones_like(x), 2*x - 1, 8*(x**2)-8*x+1, 32*(x**3)-48*(x**2)+18*x-1])

def pol_fit(xlist, ylist, basis):
    avg = 0
    for x in (ylist):
        avg += x
    avg = avg/(xlist.shape)
    lhs = basis(xlist) @ basis(xlist).T
    rhs = basis(xlist) @ ylist.T
    par = np.linalg.inv(lhs)@rhs
    return par, np.linalg.cond(lhs)

print("The condition number for the matrix while using monomials as basis is :", pol_fit(xlist, ylist, pol_basis)[1])
print("\nThe condition number for the matrix while using chebyshev polynomials as basis is :", pol_fit(xlist, ylist, cheby_basis)[1])

pol_par = pol_fit(xlist, ylist, pol_basis)[0]
cheb_par = pol_fit(xlist, ylist, cheby_basis)[0]

xlist1 = np.linspace (-0.5, 1.5, 100)
def f1(x):
    y = 0
    for i in range (0, pol_par.shape[0]):
        y += pol_par[i]*x**i
    return y

def f2(x):
    y = cheb_par[0] + cheb_par[1]*(2*x-1) + cheb_par[2]*(8*(x**2)-8*x+1) + cheb_par[3]*(32*(x**3)- 48*(x**2) + 18*x - 1) 
    return y

plt.scatter(xlist, ylist)
plt.plot(xlist1, f1(xlist1))
plt.title("Using monomials as basis")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.scatter(xlist, ylist)
plt.plot(xlist1, f2(xlist1))
plt.title("Using chebyshev polynomials as basis")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

'''
Output


The condition number for the matrix while using monomials as basis is : 12104.948671033304

The condition number for the matrix while using chebyshev polynomials as basis is : 3.856146578615564

Remark - The condition number of the matrix while using monomials as basis is quite high indicating high instability to changes in the values. Using Chebyshev polynomials is more stable. The plots with fitted functions are attached with the folder
'''