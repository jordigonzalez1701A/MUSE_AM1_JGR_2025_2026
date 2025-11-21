from numpy import zeros, linspace, log
from numpy.linalg import norm
from temporal_schemes import Cauchy_problem, Euler, Crank_Nicolson, RK4, Inverse_Euler

def richardson_extrapolation(F, U0, t1, temporal_scheme, known_q = True, **kwargs):
    """
    Estimates errors via Richardson extrapolation.
    INPUTS:
    F: Cauchy problem to integrate. U0: Initial condition. 
    U0: initial condition.
    t1: Temporal grid to know initial and final times of integration. 
    temporal_scheme: Temporal scheme. 
    known_q: Controls whether q is known (set to false to estimate q).
    OUTPUTS:
    error vector.
    """
    N1 = len(t1)
    t2 = linspace(t1[0], t1[N1-1], 2*(N1)-1)
    U_1 = Cauchy_problem(F, U0, t1, temporal_scheme, **kwargs)
    U_2 = Cauchy_problem(F, U0, t2, temporal_scheme, **kwargs)
    N2 = len(t2)
    error = zeros(N1)
    if known_q == True:
        if temporal_scheme == Euler:
            q = 1
        elif temporal_scheme == Crank_Nicolson:
            q = 2
        elif temporal_scheme == RK4:
            q = 4
        elif temporal_scheme == Inverse_Euler:
            q = 1
        else:
            print("ERROR: Temporal scheme not available.")
            return -1
        for n in range(0, N1):
            error[n] = norm(U_1[n,:]-U_2[2*n,:])/(1-1/2**q)
    else:
        for n in range(0, N1):
            error[n] = norm(U_1[n,:]-U_2[2*n,:])
    
    return error

def convergence_rate(F, U0, t1, temporal_scheme, N_ini, N_refinements=6, **kwargs):
    """ Computes the log(E) and log(N) for obtaining convergence rate of a temporal scheme. 
    INPUTS: 
    F: Cauchy problem to integrate. U0: Initial condition. 
    U0: initial condition.
    t1: Temporal grid to know initial and final times of integration. 
    temporal_scheme: Temporal scheme. 
    N_ini: Initial amount of points for the temporal scheme.
    OUTPUTS: 
    log_N: Array with the log(N) for each N 
    log_errors: Array with the log(E) for each N-grid """
    N = N_ini
    errors = zeros(N_refinements)
    log_N = zeros(N_refinements)

    t_coarse = linspace(t1[0], t1[-1], N)
    U1 = Cauchy_problem(F, U0, t_coarse, temporal_scheme, **kwargs)

    for i in range(N_refinements):

        N_fine = 2*N - 1
        t_fine = linspace(t1[0], t1[-1], N_fine)
        U2 = Cauchy_problem(F, U0, t_fine, temporal_scheme, **kwargs)

        errors[i] = norm(U2[-1,:] - U1[-1,:])
        log_N[i] = log(N_fine)

        U1 = U2
        N = N_fine

    log_errors = log(errors)
    return log_N, log_errors
        