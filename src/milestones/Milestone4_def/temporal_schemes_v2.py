from numpy import array, concatenate, zeros, linspace
from numpy.linalg import norm, solve, LinAlgError


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
    F: The function F(U,t) of the problem.
    U: The variable U of the Cauchy problem.
    t1: The initial time of the step.
    t2: The final time of the step.
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

def leapfrog(F, U, t1, t2):
    """
    Returns one step of the leapfrog method
    Inputs:
    F: The function F(U,t) of the problem.
    U: The variable U of the Cauchy problem at step n.
    t1: The initial time of the step.
    t2: The final time of the step.
    """

    N = len(U)
    a_i = F(U, t1)[N//2:N]
    v_i_plus_one_half = U[N//2:N] + 0.5*a_i*(t2-t1)
    x_i_plus_one = U[0:N//2] + v_i_plus_one_half*(t2-t1)
    a_i_plus_one = F(concatenate((x_i_plus_one,zeros(N//2))),t2)[N//2:N]
    v_i_plus_one = v_i_plus_one_half + 0.5*a_i_plus_one*(t2-t1)
 
    return concatenate((x_i_plus_one, v_i_plus_one))
