# JORDI GONZÁLEZ DE REGÀS -- Ampliación de Matemáticas 1, Máster Universitario en Sistemas Espaciales (MUSE) UPM, curso 2025-26.
# This script uses several numeric methods to solve a Kepler's initial value problem (IVP):
# r'' = - r / |r|^3, 
# r(0) = (1,0)
# r'(0) = (0,1)
# where r is the position vector, |r| is its modulus, and r'' is its second time derivative. 
# A 4-vector U is defined as the coordinates in the phase space U=(x,y,x',y'), so the equation becomes U' = F, where
# F = (r', -r/|r|^3)
# An API is provided to select the method, parameters and plot the result. The API can be applied to any dF/dt = F(U,t) Cauchy problem.


from numpy import array, concatenate, zeros, linspace
from numpy.linalg import norm, solve, LinAlgError
import matplotlib.pyplot as plt

def Euler(F, U, t1, t2, **kwargs):
    """
    Performs one step of Euler's method.
    Inputs:
    F: The function F(U,t) of the problem.
    U: The variable U of the Cauchy problem.
    t1: The initial time of the step.
    t2: The final time of the step.

    Outputs:
    U^(n+1)
    """
    return U + (t2-t1)*F(U, t1)

def Crank_Nicolson(F, U, t1, t2, tol_jacobian=None, N_max=None, newton_tol=None, **kwargs):
    """
    Performs one step of the Crank-Nicolson method.

    Inputs:
    F: The function F(U,t) of the problem.
    U: The variable U of the Cauchy problem.
    t1: The initial time of the step.
    t2: The final time of the step.
    tol_jacobian: The tolerance of the Jacobian.
    N_max: The maximum number of iterations for the Newton's method.
    newton_tol: The tolerance for Newton's method.
    """
    def crank_nicolson_target_function(U_n_plus_1, F, U_n, t1, t2):
        """
        This function is given by the definition of the Crank-Nicolson method for equations of type dU/dt = F(U),
        which is the case for Kepler's problem.
        It consists of an implicit equation that must be solved to obtain U^(n+1).
        U_n_plus_1: U^(n+1).
        U_n: U^(n).
        F: The function F such that dU/dt = F(U,t).
        t1: The initial time at the step.
        t2: The final time at the step.
        """
        delta_t = t2-t1
        return (0.5*(F(U_n_plus_1, t2) + F(U_n, t1))) - ((U_n_plus_1 - U_n)/delta_t)
    
    def Jacobian(f, x, h, F, U_n, t1, t2):
        #Computes the numeric Jacobian using symmetric differences. 
        #f: The function of which to compute the Jacobian.
        #x: The point in which to compute the Jacobian.
        #h: The tolerance (scalar).
        #args: Extra arguments for f.
        
        # Compute the dimensions of the Jacobian matrix from the function and its variable.
        n = len(x)
        m = len(f(x, F, U_n, t1, t2))

        J = zeros((m, n))
        delta_x = zeros(n)

        for i in range(0, n):
            delta_x[i] = h
            J[:, i] = (f(x+delta_x, F, U_n, t1, t2) - f(x-delta_x, F, U_n, t1, t2))/(2*h)
            delta_x[i] = 0

        return J

    def newton(f, x, h, N, newton_tol, F, U_n, t1, t2):
        """
        Performs a multidimensional Newton-Raphson method to find a root of a function.
        f: Function to solve.
        x: Target variable initial guess.
        h: tolerance for the partial derivatives (scalar).
        N: maximum number of iterations for Newton-Raphson.
        newton_tol: Tolerance for Newton-Raphson.
        """

        x_n = array(x, copy=True)
        for n in range(0, N):
            J_n = Jacobian(f, x_n, h, F, U_n, t1, t2)
            f_n = f(x_n, F, U_n, t1, t2)
            try:
                delta_x = solve(J_n, -f_n)
            except LinAlgError:
                print("LinAlgError: Newton-Raphson ill-defined.")
                break
            
            x_n = x_n + delta_x

            if(norm(delta_x)<newton_tol):
                break

        return x_n

    sol = newton(crank_nicolson_target_function, U, tol_jacobian, N_max, newton_tol, F, U, t1, t2)

    return sol

def RK4(F,U, t1, t2, **kwargs):
    """
    Performs an iteration of Runge-Kutta 4.
    U: U_n iteration of U.
    t: t_n iteration of t.
    delta_t: time distance between two time indexes.
    F: function to be solved for an equation of form dU/dt = F(U, t).
    """
    k1 = F(U, t1)
    k2 = F(U+0.5*k1*(t2-t1), t1+0.5*(t2-t1))
    k3 = F(U+0.5*k2*(t2-t1), t1+0.5*(t2-t1))
    k4 = F(U+k3*(t2-t1), t1+(t2-t1))

    return U + (1.0/6.0)*(t2-t1)*(k1 + 2*k2+2*k3 + k4)

def Inverse_Euler(F, U, t1, t2, tol_jacobian=None, N_max=None, newton_tol=None, **kwargs):

    """
    Performs one step of the inverse (backwards) Euler's method.
    Inputs:
    F: The function F(U,t) of the problem.
    U: The variable U of the Cauchy problem.
    t1: The initial time of the step.
    t2: The final time of the step.
    """
    def inverse_euler_target_function(U_n_plus_1, F, U_n, t1, t2):
        return U_n_plus_1-U_n-(t2-t1)*F(U_n_plus_1, t2)

    def Jacobian(f, x, h, F, U_n, t1, t2):
        #Computes the numeric Jacobian using symmetric differences. 
        #f: The function of which to compute the Jacobian.
        #x: The point in which to compute the Jacobian.
        #h: The tolerance (scalar).
        #args: Extra arguments for f.
        
        # Compute the dimensions of the Jacobian matrix from the function and its variable.
        n = len(x)
        m = len(f(x, F, U_n, t1, t2))

        J = zeros((m, n))
        delta_x = zeros(n)

        for i in range(0, n):
            delta_x[i] = h
            J[:, i] = (f(x+delta_x, F, U_n, t1, t2) - f(x-delta_x, F, U_n, t1, t2))/(2*h)
            delta_x[i] = 0

        return J

    def newton(f, x, h, N, newton_tol, F, U_n, t1, t2):
        """
        Performs a multidimensional Newton-Raphson method to find a root of a function.
        f: Function to solve.
        x: Target variable initial guess.
        h: tolerance for the partial derivatives (scalar).
        N: maximum number of iterations for Newton-Raphson.
        newton_tol: Tolerance for Newton-Raphson.
        """

        x_n = array(x, copy=True)
        for n in range(0, N):
            J_n = Jacobian(f, x_n, h, F, U_n, t1, t2)
            f_n = f(x_n, F, U_n, t1, t2)
            try:
                delta_x = solve(J_n, -f_n)
            except LinAlgError:
                print("LinAlgError: Newton-Raphson ill-defined.")
                break
            
            x_n = x_n + delta_x

            if(norm(delta_x)<newton_tol):
                break

        return x_n
    
    sol = newton(inverse_euler_target_function, U, tol_jacobian, N_max, newton_tol, F, U, t1, t2)

    return sol

def Cauchy_problem(F, U0, t, temporal_scheme, **kwargs):
    """
    Function to integrate a Cauchy problem using a given numeric method.
    Implicit methods will be solved using Newton's method.
    
    Inputs:
    F: F(U,t) of the Cauchy problem.
    U0: Inital condition
    t: np.array of time values on which to integrate.
    
    KWARGS:
    temporal_scheme: function to call for the numeric method.
    tol_jacobian: Tolerance of the computation of the Jacobian (if applicable).
    N_max: Max. number of iterations for Newton's method (if applicable).
    newton_tol: Tolerance for Newton's method (if applicable).
    Returns:
    U
    """
    N = len(t)
    N_v = len(U0)
    U = zeros((N, N_v))
    U[0,:] = U0

    for n in range(0, N-1):
        U[n+1,:] = temporal_scheme(F, U[n,:], t[n], t[n+1], **kwargs)

    return U

def F(U, t):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    Note: F is not explicitly dependent on t in this case, adding it allows to construct
    robust functions for the numerical methods.
    """
    r = U[0:2]
    rd = U[2:4]

    return concatenate((rd, -r/norm(r)**3), axis=None)

def plot_orbit(U, method, N, T, delta_t):
    plt.axis("equal")
    plt.plot(U[:, 0], U[:, 1], label=f"N={N}, delta_t={delta_t}")
    plt.scatter(U[:, 0], U[:, 1], color='teal', s=2, label="Time steps")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.text(x=0.8, y=0.8, s=f"{method}:\n T={T},\nN={N},\ndelta_t={delta_t:.2e}")
    plt.show()
    return

U0 = array([1, 0, 0, 1])
T = 10
N = 20000
t = linspace(0, T, N)
delta_t = (t[1]-t[0])/N
U = Cauchy_problem(F, U0, t, Inverse_Euler, tol_jacobian=1e-10, N_max=10000, newton_tol=1e-10)

plot_orbit(U, "Inverse Euler", N, T, delta_t)