from numpy import array, concatenate, zeros, linspace, log
from numpy.linalg import norm, solve, LinAlgError
import matplotlib.pyplot as plt
from temporal_schemes import Cauchy_problem, Euler, Crank_Nicolson, RK4, Inverse_Euler
from differential_equation import F

def richardson_extrapolation(F, U0, t1, temporal_scheme, **kwargs):
    N1 = len(t1)
    t2 = linspace(t1[0], t1[-1], 2*(N1-1)-1)
    U_1 = Cauchy_problem(F, U0, t1, temporal_scheme, **kwargs)
    U_2 = Cauchy_problem(F, U0, t2, temporal_scheme, **kwargs)
    N2 = len(t2)

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
    error = norm(U_1[N1-1,:]-U_2[N2-1,:])/(1-1/2**q)
    return error

def convergence_rate(F, U0, t1, temporal_scheme, **kwargs):
    N1 = len(t1)
    t2 = linspace(t1[0], t1[-1], 2*(N1-1)+1)
    t3 = linspace(t1[0], t1[-1], 4*(N1-1)+1)
    tref = linspace(t1[0], t1[-1], 20*(N1-1)+1)
    U_1 = Cauchy_problem(F, U0, t1, temporal_scheme, **kwargs)
    U_2 = Cauchy_problem(F, U0, t2, temporal_scheme, **kwargs)
    U_3 = Cauchy_problem(F, U0, t3, temporal_scheme, **kwargs)
    U_ref = Cauchy_problem(F, U0, tref, temporal_scheme, **kwargs)
    E_1 = norm(U_1[-1,:]-U_ref[-1,:])
    E_2 = norm(U_2[-1,:]-U_ref[-1,:])
    E_3 = norm(U_3[-1,:]-U_ref[-1,:])
    delta_t1 = (t1[1]-t1[0])
    delta_t2 = t2[1]-t2[0]
    delta_t3 = t3[1]-t3[0]
    p_12 = log(E_1/E_2)/log(delta_t1/delta_t2)
    p_23 = log(E_2/E_3)/log(delta_t2/delta_t3)
    print(f"p12: {p_12}, p23: {p_23}")
    return (p_12+p_23)/2

U0 = array([1, 0, 0, 1])
T = 10
N = 30000
t1 = linspace(0, T, N)
error_rk4 = richardson_extrapolation(F, U0, t1, Inverse_Euler, tol_jacobian=1e-9, N_max=10000, newton_tol=1e-9)
conv_rate_rk4 = convergence_rate(F, U0, t1, Inverse_Euler, tol_jacobian=1e-9, N_max=10000, newton_tol=1e-9)
print(f"error RK4: {error_rk4:.5f}")
print(f"conv rate RK4: {conv_rate_rk4}")