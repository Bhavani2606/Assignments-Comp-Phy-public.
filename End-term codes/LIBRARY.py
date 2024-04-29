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
    
