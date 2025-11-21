from cauchy_problem import Cauchy_problem
import matplotlib.pyplot as plt
from numpy import abs

def plot_CRTBP_orbits(problem, U0_list, t, time_scheme, N_body=2, **kwargs):
    """
    Plots orbits with multiple initial conditions on a single plot.
    problem: The cauchy problem to solve.
    U0_list: List of initial conditions
    t: Time values for the integration (array).
    time_scheme: Time scheme for integration.
    """
    tol = 1e-9
    
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)     
        U_plot = U.copy()
        U_plot[abs(U_plot) < tol] = 0
        plt.plot(U_plot[:,0], U_plot[:,1])
        
    plt.title(f"Posiciones del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return

def plot_CRTBP_velocities(problem, U0_list, t, time_scheme, N_body=2, **kwargs):
    """
    Plots orbits with multiple initial conditions on a single plot.
    problem: The cauchy problem to solve.
    U0_list: List of initial conditions
    t: Time values for the integration (array).
    time_scheme: Time scheme for integration.
    """
    ax = plt.figure().add_subplot(projection='3d')
    dim = 3
    tol = 1e-9
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)
        U_plot = U.copy()
        U_plot[abs(U_plot) < tol] = 0
        ax.plot(U_plot[:,3], U_plot[:,4], U_plot[:,5])
        
    ax.set_title(f"Velocidades del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_zlabel("v_z")
    plt.show()
    return