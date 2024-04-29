# to make a library of all the codes required.

import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.special as sp
import scipy.optimize as opt


'''
-------------------------------------------------------Root finding methods-------------------------------------------------------------------
'''

#to implement Newton-Raphson method to find the roots of an equation. function requires the functin to be solved and an initial guess
def newtraph(x, func, check = 0.0001):
    f, df = func(x)
    if abs(df) < 0.01:
        print("take new guess") 
        sys.exit
    x1 = x + 12
    rootlist = [x]
    count = 0
    while count <= 10000 and abs(f) > check:
        rat = f/df
        x1 = x
        x = x - rat
        f, df = func(x)
        rootlist.append(x)
        count += 1
    return x, count, rootlist


def bracket(x, func):# to implement bracketting for bisection method. requires two values of x and the function
    y = [func(x[0]), func(x[1])]#dummy range of function values
    iter = 0#to check the number of iterations
    while (y[0]*y[1]) >= 0 and iter <=100:#to run the loop until the functions have opposite signs or until the iterations have 
          
        if abs(y[1]) <= abs(y[0]):#to check which endpoint's function value is closer to zero
            x[1] = x[1] + (0.5*(x[1] - x[0]))#changing the endpoints accordingly.
        else:
            x[0] = x[0] - 0.5*(x[1] - x[0])
        y[0] = func(x[0])#to find the function values of the end points of the interval
        y[1] = func(x[1]) 
        iter += 1
    return x

# print(bracket([1.5, 2.5]))

def bisection(xinit, check, func):#to implement the bisection method to find roots.
    x = bracket(xinit, func)#to have proper bracket
    iter = 0
    rootlist = []
    while abs(x[0]-x[1])>check and abs(func(x[0])) > check and abs(func(x[1])) > check and iter < 1000:#repeating iterations until conditions are satisfied
        c = (x[0] + x[1])/2
        y = func(c)
        rootlist.append(c)
        if func(c)*func(x[0]) < 0:#if function at c and x[0] are opposite, then the interval is changed
            x[1] = c
        elif func(c)*func(x[1]) < 0:#if function at c and x[0] are opposite, then the interval is changed
            x[0] = c
        else:
            continue
        #print("Iteration number = ", iter, " function value = ", func(c))
        iter += 1
    if func(x[0]) == 0:#is f(x[0]) is zero
        print("0 iterations")
        return x[0]
    elif func(x[1]) == 0:
        print("0 iterations")
        return x[1]
    else:
        return c, iter, rootlist

def reg(xinit, eps, delt, func):# to implement regula falsi method to find the roots. Args - initial x values for bracketting, tolerance for the consecutive iteration roots, tolerance for the funcitons at roots fro consecutive iterations, function.
    x = bracket(xinit)#to implement bracketing to get proper interval
    y = [func(x[0]), func(x[1])]
    count = 0
    rootlist = []
    while abs(y[0])>eps and abs(y[1])>eps and (abs(x[0] - x[1]) >= delt) and count <= 1000:#iterations will continue until conditions are met
        y[0] = func(x[0])
        y[1] = func(x[1])
        # print("y0 = ", y[0])
        # print("y1 = ", y[1])
        c = x[1] - ((x[1] - x[0])*y[1])/(y[1] - y[0])#finding the next point
        fc = func(c)
        rootlist.append(c)
        # print("c=", c)
        # print("fc=", fc)
        if fc*y[0] < 0:#changing the interval
            x[1] = c
        elif fc*y[1] < 0:
            x[0] = c
        else:
            sol = c
            break
        # print("x1=", x[1])
        # print("x0=", x[0])
        #print ("Iteration number = ", count, "; Function = ", func(c))
        count += 1
    sol = c
    return sol, count, rootlist

def fixpt(guess, err, func):#this function takes the guess and the permissible error as the input and finds the root using the fixed point method.
    x = guess
    x1 = x+3 # decalaring another variable for storage
    iter = 0 #to keep count of the number of iterations
    while abs(x - x1) >= err:
        x1 = x
        x = func(x)
        iter += 1
    return x, iter

'''
-------------------------------------------------------End of root finding methods----------------------------------------------------------------------
'''

'''
-------------------------------------------------------Numerical Quadrature---------------------------------------------------------------------
'''

def ceil(p):#to implement the ceiling funciton
    if p == int(p):
        return p
    else:
        s = int(p)
        return s+1

        
def upbound(bound, l, u, d2f, d4f, str):#to find the number of steps required to achieve a given tolerance, for a given algorithm
    mpt = math.sqrt((((u-l)**3)*d2f)/(24*bound))#for number of intervals
    trp = math.sqrt((((u-l)**3)*d2f)/(12*bound))
    sim = ((((u-l)**5)*d4f)/(180*bound))**(0.25)
    if str == "midpoint":
        return ceil(mpt)
    elif str == "trapezoid":
        return ceil(trp)
    elif str == "simpson":
        return ceil(sim)

def trapezoid(l, u, N, function1):#to implement the trapezoid method to find the integral of a given function. Args - lower bound, upper bound,  number of steps, function to be integrated
    p = abs(u-l)/N
    q = l
    sum1 = 0
    for i in range (0, N):
        l1 = function1(q)
        u1 = function1(q+p)
        sum1 += (l1 + u1)*p/2
        q += p
    return sum1

def midpoint(l, u, N, function1): # function1 is function, u is upper and l is the lower limit
    p = abs(u-l)/N#to determine the length of the interval
    q = l
    sum = 0#to store the total sum
    for i in range (0, N):
        midpt = q + (p/2)#midpoint of the interval considered
        fmp = function1(midpt)#function at the midpoint
        sum += (fmp*p)
        q += p
    return sum

def simp(l, u, N, function):#to implement the simpson method
    h = (u - l)/(N)
    x0 = l
    x2 = l + h
    sum1 = 0
    for i in range (0, N):
        x1 = x0 + (h/2)
        f0 = function(x0)
        f1 = function(x1)
        f2 = function(x2)
        sum1 += f0 +(4*f1) + f2
        x0 += h
        x2 += h
    integ = sum1*(h/6)
    return integ


# Function to get the Gauss-Legendre nodes and weights
def gauss_legendre(n):
    """
    Get the nodes and weights for Gauss-Legendre quadrature.
    n: the number of points
    """
    # Roots and weights of Legendre polynomial
    roots, weights = sp.roots_legendre(n)
    return roots, weights

# Function to compute the integral using Gaussian quadrature
def gaussian_quadrature(func, a, b, n):
    """
    Approximate the integral of 'func' from 'a' to 'b' using Gaussian quadrature.
    func: the function to integrate
    a: the start of the interval
    b: the end of the interval
    n: the number of quadrature points
    """
    # Get the nodes and weights for Gauss-Legendre quadrature
    roots, weights = gauss_legendre(n)
    
    # Map roots from [-1, 1] to [a, b]
    x = 0.5 * ((b - a) * roots + (b + a))
    
    # Calculate the function values at the mapped points
    fx = np.array([func(xi) for xi in x])
    
    # Approximate the integral using the mapped nodes and weights
    integral = 0.5 * (b - a) * np.dot(weights, fx)
    
    return integral

'''
---------------------------------------------------End of numerical integration values------------------------------------------------------------------------
'''

'''
----------------------------------------------------Differential equation solver---------------------------------------------------------------------------
'''
def forward_euler(ode, y0, t0, t_end, dt):#ode is the derivative of y wrt t, y0, x0 are the initital boundary conditions, t_end is the end time, dt is the time skip
    t = np.arange(t0, t_end + dt, dt)# Time array
    # Initialize array for solutions with initial condition
    y = np.zeros_like(t)
    y[0] = y0  # initial condition

    # Apply the Forward Euler method
    for i in range(1, len(t)):
        y[i] = y[i - 1] + dt * ode(t[i - 1], y[i - 1])

    return t, y

def backward_euler(ode, y0, t0, t_end, dt):

    t = np.arange(t0, t_end + dt, dt)# Time array
    # Initialize array for solutions with initial condition
    y = np.zeros_like(t)
    y[0] = y0  # initial condition

    for i in range (0, len(y)):
        def implicit_eq(y_new):
            return y_new - y[i - 1] - dt * ode(t[i], y_new)

        # Solve for y_new
        y[i] = opt.fsolve(implicit_eq, y[i - 1])[0]

    return t, y

def predictor_corrector(ode, y0, t0, t_end, dt):
    t = np.arange(t0, t_end + dt, dt)# Time array
    # Initialize array for solutions with initial condition
    y = np.zeros_like(t)
    y[0] = y0  # initial condition

    for i in range(1, len(y)):
        y[i] = y[i-1] + dt*ode(t[i-1], y[i-1])
        y[i] = y[i-1]+dt*(ode(t[i-1], y[i-1])+ode(t[i], y[i]))/2
    return t, y

def RK_4 (x, y, h, xf, func):# to  implement RK4 method to find the solution of a differential equation.
    x_val = [x]
    y_val = [y]
    x1 = x
    y1 = y
    while x < xf:
        k1 = h*func(x, y)
        k2 = h*func((x+0.5*h), (y+0.5*k1))
        k3 = h*func((x+0.5*h), (y+0.5*k2))
        k4 = h*func((x+h), y+k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += (h)
        x_val.append(x)
        y_val.append(y)
    
    return x_val, y_val

def RK_4_multi(x_init, y_init, h, x_final, func):#to implement RK4 to solve n coupled ode. xinit, yinit, h, x_final are all first rank arrays of length n, function must take the array of x values and return the arrray of y values
    n = len(y_init)
    y_init = np.array(y_init)
    x_val = np.arange(x_init, x_final, h)
    m = len(x_val)
    y_val = np.zeros((n, m))
    x_val[0] = x_init
    y_val[:, 0] = y_init[:]
    for i in range (1, m):
        k1 = k2 = k3 = k4 = np.zeros(n)
        k1 = h*func(x_val[i-1], y_val[:, i-1])
        k2 = h*func((x_val[i-1]+0.5*h), (y_val[:, i-1]+0.5*k1))
        k3 = h*func((x_val[i-1]+0.5*h), (y_val[:, i-1]+0.5*k2))
        k4 = h*func((x_val[i-1]+h), y_val[:, i-1]+k3)
        y_val[:, i] = y_val[:, i-1]+(k1 + 2 * k2 + 2 * k3 + k4) / 6
    # print(x_val, y_val)
    
    return x_val, y_val

def velocity_verlet(position, velocity, acceleration, dt):#to implement the velocity verlet algorithm to ypdate the posiiton and velocity of an object under Newton's laws of motion. Args - initial position and velocity, acceleration and time step.
    # Half-step update for velocity
    velocity_half = velocity + 0.5 * acceleration * dt
    # Full-step update for position
    position += velocity_half * dt
    # Compute new acceleration based on the updated position
    new_acceleration = acceleration(position)
    # Full-step update for velocity
    velocity = velocity_half + 0.5 * new_acceleration * dt
    return position, velocity, new_acceleration

def vel_ver_array(pos_init, vel_init, acceleration, time_init, time_fin, dt):# to implement the velocity verlet multiple times and get the full array. acceleration had to only depend on position.
    position = pos_init
    velocity = vel_init
    position_arr = []
    vel_arr = []
    time = np.arange(time_init, time_fin, dt)
    for i in time:
        position_arr.append(position)
        vel_arr.append(velocity)
        position, velocity, acceleration = velocity_verlet(position, velocity, acceleration(position))

def leapfrog (pos_init, vel_init, acceleration, t0, t_fin, dt): #to implement the leapfrog algorithm to simulatte the dynamics of an object under force. Args - initial position, initial momentum, acceleration, initial and final time, and step size.
    #function returns the position array, velocuty arrayn and time stamps where the positions and velocitites are calculated...
    position = pos_init
    velocity = vel_init
    pos_arr = []
    vel_arr = [vel_init]
    velocity += acceleration(position)*(dt/2)
    time_arr_pos = np.arange(t0, t_fin, dt)
    time_arr_vel = np.arange(t0+(dt/2), t_fin-(dt/2), dt)

    for i in range (len(time_arr_vel)-1):
        vel_arr.append(velocity)
        pos_arr.append(position)
        position += velocity*dt
        velocity += acceleration(position)*(dt)
    pos_arr.append(position)
    vel_arr.append(velocity)
    position += velocity*dt
    velocity += acceleration(position)*(dt/2)
    pos_arr.append(position)
    vel_arr.append(velocity)
    time_arr_vel = np.append(t0, time_arr_vel)
    time_arr_vel = np.append(time_arr_vel, t_fin)
    return pos_arr, vel_arr, time_arr_pos, time_arr_vel

def shooting (x, bv, func, guess1, guess2, tolerance = 0.001):#to implement shooting method to solve second order differential equation. x ocntains the initial and final values of x, bv are the boundary values at the initial and final values of x, funciton returns the differential equations. guess1 and guess2 are the two initila guess values.
    sol1 = RK_4_multi(x[0], [bv[0], guess1], h = 0.001, x_final=x[1], func = func)[1][0]
    sol2 = RK_4_multi(x[0], [bv[0], guess2], h = 0.001, x_final=x[1], func = func)[1][0]
    count = 0
    # print(sol2[-1]-bv[1])
    while abs(sol2[-1] - bv[1]) > tolerance:
        # print(sol2[-1])
        guess3 = guess2 + (guess1 - guess2)*(bv[1] - sol2[-1])/(sol1[-1] - sol2[-1])
        guess1 = guess2
        guess2 = guess3
        sol1 = RK_4_multi(x[0], [bv[0], guess1], h = 0.001, x_final=x[1], func = func)[1][0]
        sol2 = RK_4_multi(x[0], [bv[0], guess2], h = 0.001, x_final=x[1], func = func)[1][0]
        count+=1
    sol = RK_4_multi(x[0], [bv[0], guess2], h = 0.001, x_final=x[1], func = func)
    # print(count)
    return sol
    
'''
-------------------------------------------------------End of differential equation codes----------------------------------------------------------------------
'''

'''
-------------------------------------------------------Matrices and system of linear equations---------------------------------------------------------
'''

#To implement Gauss-Jacobi method to solve the system of linear equations
def jacobi(a, b, error1):# a is the coefficient matrix, b is the rhs matrix.
    x = []#kth solution
    y = []#k+1th solution
    diff = []#for checking precision
    for p in range (0, len(a)):
        x.append([1])
        y.append([1])
        diff.append([100])
    check = 1 #to check if we need further iterations
    iter_max = 0 #to count the number of iterations
    while check == 1 and iter_max < 25: #while loop for iterations
        for i in range (0, len(a)):
            if diff[i][0] > error1:#to check if any solution falls beyond the permissible error
                check = 1
                break
            else:
                check = 0
        
        for k in range (0, len(a)):#for all the variables in the i_th solution vector
            sum = 0
            for l in range (0, len(a)):# to find a_ij*x_j
                if l != k:
                    sum += a[k][l]*x[l][0]
            y[k][0] = (b[k][0] - sum)/a[k][k] #to determine the i+1_th solution
        iter_max += 1
        for n in range (0, len(a)):
            diff[n][0] = abs(y[n][0] - x[n][0])
        for i in range (0, len(x)):
            x[i][0] = y[i][0]
    return y, iter_max, diff

def rowswap(a, b):#to swap two rows
    for i in range (0, len(a)):
        c = a[i]
        a[i] = b[i]
        b[i] = c
def rowmult(a, b): #to multiply row a with number b
    for i in range (0, len(a)):
        a[i] = a[i]*b
def rowsub(a, b, d):# to subtract multiple of b from a, a - cb 
    c = []
    for i in range (0, len(a)):
        c.append(a[i] - ((b[i])*d))
    return c

def gj(a, rhs):
    for p in range (0, len(rhs)):#to have the lhs and the rhs in one matrix
        a[p].append(rhs[p][0])
    check = 1
    for i in range (0, len(a)):
        if a[i][i] == 0:#to check if the pivot element is non-zero
            check = 0
            for j in range (i+1, (len(a))):#to swap the rows so that the leading element is non-zero
                if a[j][i] != 0:
                    rowswap(a[i], a[j])
                    check = 1
                    break
        if check == 0:
            print("unique solution does not exist.")
            sys.exit()
        rowmult(a[i], (1/a[i][i]))#to divide the row bu the leading element
        for k in range (0, len(a)):
            if k != i:
                a[k] = rowsub(a[k], a[i], a[k][i])#to make the non-leading terms of row 0
    sol = []
    for l in range (0, len(a)):#to extract the last column of the matrix as the solution
        sol.append(a[l][len(a)])
    return sol

#To implement Gauss-Siedel method to solve system of linear equations
def gs(a, b, check1):#a is the coefficientmatrix, b is the rhs matrix, check is the permissible error.
    norm = 3
    max_iter = 0
    x=[]
    for i in range (0, len(b)):
        x.append([2])
    while max_iter < 100 and norm > check1:
        y = []
        for p in range (0, len(a)):
            y.append([x[p][0]])
        #print(y)
        for k in range (0, len(a)):#for each variable
            sum1 = 0
            j = 0
            while j < len(a):#to find the sum in the formula
                if j != k:
                    #print(j, x[j][0])
                    sum1 += a[k][j]*x[j][0]
                j+=1
            #print(sum1)
            c = (b[k][0] - sum1)/a[k][k]
            x[k][0] = c
            #print(x)
        norm = 0
        for l in range (0, len(x)):#to check the difference in the previous and the next set of values
            norm += (x[l][0] - y[l][0])**2
        #print(y)
        norm = norm**0.5
        max_iter +=1
    if max_iter == 100:
        print("maximum iterations reached")
    return x, max_iter, norm  

#To solve the system of linear equations using LU_decomposition

def rowswap_LU(a, b):
    for i in range (0, len(a)):
        c = a[i]
        a[i] = b[i]
        b[i] = c
    
def LU(mat, rhs):
    if mat[0][0] == 0:#to make the first diagonal element  non-zero
        for i in range (1, len(mat)):
            if mat[i][0] != 0:
                rowswap_LU(mat[i], mat[0])
                rowswap_LU(rhs[i], rhs[0])
                break
    upper = []
    lower = []        
    for p in range (0, len(mat)):#to create the upper and the lower matrices
        c = []
        for q in range (0, len(mat)):
            c.append(mat[p][q])
        upper.append(c)

    for p in range (0, len(mat)):
        c = []
        for q in range (0, len(mat)):
            c.append(mat[p][q])
        lower.append(c)  

    
    # print("u - ", upper)
    # print(lower)
    for j in range (0, len(mat)):
        for i in range (0, len(mat)):
            if i < j:#for the elements in upper triangle of both the matrices
                lower[i][j] = 0
                sum = 0
                k = 0
                while k <= i-1:
                    sum += (upper[k][j]*lower[i][k])
                    k += 1
                #print("c1:", sum)
                upper[i][j] = mat[i][j] - sum
            elif i == j:#for the diagonal elements of both the matrices
                sum = 0
                k = 0
                while k <= i -1:
                    sum += (lower[i][k]*upper[k][j])
                    k += 1
                #print("c2", sum)
                #print("ele", mat[i][j])
                upper[i][j] = mat[i][j] - sum
                lower[i][j] = 1
            else:#for the lower triangle matrix of both the matrices
                sum = 0
                k = 0
                while k <= j-1:
                    sum += (lower[i][k]*upper[k][j])
                    k += 1
                #print("c3", sum)
                lower[i][j] = (mat[i][j] - sum)/upper[j][j]
                upper[i][j] = 0
    for i in range(0, len(upper)):
        for j in range (0, len(upper)):
            upper[i][j] = round(upper[i][j], 10)
            lower[i][j] = round(lower[i][j], 10)
    return upper, lower        

def chol(a):
    for i in range (0, len(a)):#to check if the matrix is symmetric
        for j in range (0, len(a)):
            if a[i][j] != a[j][i]:
                print("matrix is not symmetric")
                sys.exit
    L=a
    for i in range (0, len(a)):
        for j in range (0, len(a)):
            if i == j:#for the diagonal elements
                sum = 0
                k = 0
                while k <= i-1:
                    sum += L[k][i]**2
                    k += 1
                L[i][j] = math.sqrt(a[i][j] - sum)
            elif i<j:#for the upper triangle elements
                sum = 0
                k = 0
                while k <= i-1:
                    sum += L[k][i]*L[k][j] 
                    k += 1
                L[i][j] = (a[i][j] - sum)/L[i][i]
            else:#for the lower triangle elements
                L[i][j] = 0
    return L

def for_sub(a, b):#a is the coefficient matrix and b is the rhs matrix
    sol = []
    for i in range (0, len(b)):#to apply for each variable 
        sum = 0
        for j in range (0, i):
            sum += sol[j]*a[i][j]
        c = b[i][0] - sum
        sol.append(c/a[i][i])
    sol1 = []
    for i in range (0, len(sol)):#to make a column matrix
        c = [sol[i]]
        sol1.append(c)
    return sol1

#to implement backward substitution(upper triangle)
def back_sub(a, b):
    i = len(a)-1
    sol = []
    for p in range (0, len(a)):#creating initial matrix
        sol.append(0)
    while i >= 0:#applying the formula, starting from the last element
        sum = 0
        for j in range (i+1, len(a)):
            sum += a[i][j]*sol[j]
        sol[i] = (b[i][0] - sum)/a[i][i]
        i -= 1
    sol1 = []
    for i in range (0, len(sol)):#making it a column matrix
        c = [sol[i]]
        sol1.append(c)
    return sol1

#to implement conjugate gradient method to find the inverse of a matrix
def conj_grad(A, b, x0, tol, max_iter=1000):
    '''A : The coefficient matrix.
        b : The rhs vector.
        x0 : The initial guess
        tol : Tolerance for the residual norm.
        max_iter : Maximum number of iterations.'''
    x = np.array(x0)
    r = b - np.dot(A, x)
    d = r
    r_normold = np.dot(r.transpose(), r)

    for iterations in range(1, max_iter + 1):
        Ad = np.dot(A, d)
        alpha = r_normold / np.dot(d.transpose(), Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        r_normnew = np.dot(r.transpose(),r)
        if np.sqrt(r_normnew) < tol:
            break
        d = r + (r_normnew / r_normold) * d
        r_normold = r_normnew
    return np.array(x), iterations
def conj_matinv(A):#to calculate the inverse of a matrix using conjugate gradient method.
    n = len(A)    
    B = np.eye(n)
    r = np.zeros((n,n))
    c = np.zeros((n,n))
    for i in range (n):
        r[:, i] = conj_grad(A, B[:, i], c[:, i], 1e-7)[0]
    return r

#to implement the QR factorization using Gram Schmidt orthogonalization.
def QR(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j]*Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v/R[j, j]
    return Q, R

# Function to create a Givens rotation matrix
def givens_rotation(i, j, theta, n):
    """
    Create a Givens rotation matrix for zeroing out the (i, j)th element.
    
    Parameters:
    - i, j: Indices for the rotation
    - theta: Rotation angle
    - n: Dimension of the rotation matrix
    
    Returns:
    - G: Givens rotation matrix of size (n, n)
    """
    G = np.eye(n)
    G[i, i] = np.cos(theta)
    G[j, j] = np.cos(theta)
    G[i, j] = -np.sin(theta)
    G[j, i] = np.sin(theta)
    return G

# Function to perform QR decomposition using Givens rotations
def qr_decomposition(A):
    """
    Perform QR decomposition using Givens rotations.
    
    Parameters:
    - A: Square matrix
    
    Returns:
    - Q, R: QR decomposition
    """
    n = A.shape[0]
    Q = np.eye(n)  # Initialize Q as the identity matrix
    R = A.copy()  # Copy of A to modify
    
    # Apply Givens rotations to zero out elements below the diagonal
    for j in range(n):
        for i in range(j + 1, n):
            if R[i, j] != 0:
                # Calculate the angle for Givens rotation
                r = np.hypot(R[j, j], R[i, j])  # Magnitude of the resultant vector
                c = R[j, j] / r  # Cosine of the rotation angle
                s = -R[i, j] / r  # Sine of the rotation angle
                
                # Create the Givens rotation matrix
                G = givens_rotation(j, i, np.arccos(c), n)
                
                # Apply Givens rotation to R and update Q
                R = G @ R
                Q = Q @ G.T
    
    return Q, R

# Function to find eigenvalues using the QR algorithm with Givens rotations
def qr_algorithm(A, iterations=100):
    """
    Find eigenvalues using the QR algorithm with Givens rotations.
    
    Parameters:
    - A: Square matrix
    - iterations: Number of QR iterations
    
    Returns:
    - eigenvalues: List of eigenvalues
    """
    n = A.shape[0]
    B = A.copy()  # Make a copy of the matrix
    
    # Perform QR iterations
    for _ in range(iterations):
        Q, R = qr_decomposition(B)
        B = R @ Q
    
    # Eigenvalues are along the diagonal
    eigenvalues = np.diag(B)
    return eigenvalues

def pow_iter(A, num_iterations=1000, tol=1e-6):#to implement power iteraitons to find the largest eigenvalue and the corresponding eigenvector
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
'''
-------------------------------------------------end of matrices and system of linear equations----------------------------------------------------------------
'''

'''
-------------------------------------------------Statistics and fitting-----------------------------------------------------------------------------------------------
'''

def pol_fit(xlist, ylist, basis):#to return the parameters of polynomial fit and the condition number of the matrix
    avg = 0
    for x in (ylist):
        avg += x
    avg = avg/(xlist.shape)
    lhs = basis(xlist) @ basis(xlist).T
    rhs = basis(xlist) @ ylist.T
    par = np.linalg.inv(lhs)@rhs
    return par, np.linalg.cond(lhs)
