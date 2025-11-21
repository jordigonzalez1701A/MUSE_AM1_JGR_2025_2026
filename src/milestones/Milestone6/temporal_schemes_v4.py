from numpy import array, concatenate, zeros, linspace, dot, sum
from numpy.linalg import norm, solve, LinAlgError
from numpy import sum as npsum


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

def single_ERK_step(F, U, t1, t2, e=None, butcher_array=None, b=None, c=None, reuse_k = False, k=None, **kwargs):
    h = t2-t1
    if reuse_k == False:
        ks = zeros((e, len(U)))
        U_stage = U.copy()
        for i in range(0, e):
            for j in range(0,i):
                U_stage += h*butcher_array[i,j]*ks[j]

            
            k_i = F(U_stage, t1 + c[i]*h, **kwargs)

            ks[i] = k_i

        U_n_plus_one = U + h *dot(b, ks)
        return U_n_plus_one, ks
    else:
        U_n_plus_one = U + h*dot(b,k)
        return U_n_plus_one, k

    

def ERK_stepsize(U1, U2, ERK_step_tol, q, h):
    T_n_plus_one = norm(U1-U2)
    if T_n_plus_one > ERK_step_tol:
        return h*(ERK_step_tol/T_n_plus_one)**(1/(q+1))
    else:
        return h

def embedded_RK(F, U, t1, t2, e=None, butcher_array=None, b_1=None, b_2=None, c=None, q=None, ERK_tol=None, **kwargs):
    h = t2 - t1
    U_n_plus_one_1, k_vals = single_ERK_step(F, U, t1, t2, e=e, butcher_array=butcher_array, b=b_1, c=c, reuse_k=False)
    U_n_plus_one_2, k_vals = single_ERK_step(F, U, t1, t2, e=e, butcher_array=butcher_array, b=b_2, c=c, reuse_k=True, k=k_vals)

    h = ERK_stepsize(U_n_plus_one_1, U_n_plus_one_2, ERK_tol, q, h)
    N_t = int((t2-t1)/h)+1
    U_i = U.copy()

    for i in range(0, N_t-1):
        U_n_plus_one,_ = single_ERK_step(F, U_i, t1+i*h, t1+(i+1)*h, e=e, butcher_array=butcher_array, b=b_1, c=c, reuse_k=False, **kwargs)
        U_i[:] = U_n_plus_one[:]

    return U_i

def Butcher_tableau(method_name):
    if method_name == "Dormand-Prince":
        e = 7
        c = array([0.0, 0.2, 0.3, 0.8, 8/9, 1.0, 1.0])
        a = zeros((e,e))
        a = array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
             [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
             [3/40, 9/40, 0.0, 0.0, 0.0, 0.0, 0.0], 
             [44/45, -56/15, 32/9, 0.0, 0.0, 0.0, 0.0], 
             [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0, 0.0], 
             [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0, 0.0], 
             [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0] ])
        b1 = array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
        b2 = array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        q = 5
    elif method_name == "RKF45":
        e = 6
        c =array([0.0, 1/4, 3/8, 12/13, 1.0, 1/2])
        a = array([ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [1/4, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [3/32, 9/32, 0.0, 0.0, 0.0, 0.0], 
                   [1932/2197, -7200/2197, 7296/2197, 0.0, 0.0, 0.0], 
                   [439/216, -8.0, 3680/513, -845/4104, 0.0, 0.0], 
                   [-8/27, 2.0, -3544/2565, 1859/4104, -11/40, 0.0] ])
        b1 = array([25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0])
        b2 = array([16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55])
        q = 4
    elif method_name == "Cash-Karp":
        e = 6
        c = array([0.0, 1/5, 3/10, 3/5, 1.0, 7/8])
        a = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [1/5, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [3/40, 9/40, 0.0, 0.0, 0.0, 0.0], 
                   [3/10, -9/10, 6/5, 0.0, 0.0, 0.0], 
                   [-11/54, 5/2, -70/27, 35/27, 0.0, 0.0], 
                   [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0.0] ])
        b1 = array([37/378, 0.0, 250/621, 125/594, 0.0, 512/1771])
        b2 = array([2825/27648, 0.0, 18575/48384, 13525/55296, 277/14336, 1/4])
        q = 5
    else:
        print("ERROR: Invalid embedded RK method selected")
        return

    return e, a, b1, b2, c, q


    
        
    
            

    