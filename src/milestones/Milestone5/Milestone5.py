from physics_v3 import N_body_problem
from temporal_schemes_v3 import Euler, RK4, Crank_Nicolson
from cauchy_problem import Cauchy_problem
from numpy import linspace, array
from matplotlib import pyplot as plt

def plot_N_body_problem_orbits(problem, U0_list, t, time_scheme, N_body=5, **kwargs):
    """
    Plots orbits with multiple initial conditions on a single plot.
    problem: The cauchy problem to solve.
    U0_list: List of initial conditions
    t: Time values for the integration (array).
    time_scheme: Time scheme for integration.
    """
    ax = plt.figure().add_subplot(projection='3d')
    dim = 3
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)
        #plt.scatter(U[:, 0], U[:, 1], color='teal', s=1, label="Time steps")
        for i in range(0, N_body):
            ax.plot(U[:,2*i*dim], U[:,2*i*dim+1], U[:,2*i*dim+2], label=f"body {i}")
        #plt.text(x=2.4, y=2.3, s=f"{time_scheme.__name__}:\n T={t[-1]},\nN={len(t)},\ndelta_t={T/len(t):.2e}")
    ax.set_title(f"Posiciones del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()
    return

def plot_N_body_problem_velocities(problem, U0_list, t, time_scheme, N_body=5, **kwargs):
    """
    Plots orbits with multiple initial conditions on a single plot.
    problem: The cauchy problem to solve.
    U0_list: List of initial conditions
    t: Time values for the integration (array).
    time_scheme: Time scheme for integration.
    """
    ax = plt.figure().add_subplot(projection='3d')
    dim = 3
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)
        #plt.scatter(U[:, 0], U[:, 1], color='teal', s=1, label="Time steps")
        for i in range(0, N_body):
            ax.plot(U[:,2*i*dim+dim], U[:,2*i*dim+dim+1], U[:,2*i*dim+dim+2], label=f"body {i}")
        #plt.text(x=2.4, y=2.3, s=f"{time_scheme.__name__}:\n T={t[-1]},\nN={len(t)},\ndelta_t={T/len(t):.2e}")
    ax.set_title(f"Velocidades del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_zlabel("v_z")
    ax.legend()
    plt.show()
    return

T = 10
N_T = 1000
t = linspace(0, T, N_T)
U0 = array([
    # r0
    -1.0,  0.5,  0.0,
    # v0
     0.0,  0.3,  0.1,
    # r1
     1.2, -0.4,  0.3,
    # v1
    -0.2,  0.1,  0.0,
    # r2
     0.0,  1.5, -0.2,
    # v2
     0.1, -0.1,  0.2,
    # r3
    -1.3, -1.0,  0.4,
    # v3
     0.2,  0.0, -0.1,
    # r4
     0.8,  0.9, -0.5,
    # v4
    -0.1, -0.2,  0.1,
])
U0_list = [U0]
tol_jacobian = 1e-9
N_max = 10000
newton_tol = 1e-10
plot_N_body_problem_orbits(N_body_problem, U0_list, t, Crank_Nicolson, tol_jacobian=tol_jacobian, newton_tol=newton_tol, N_max=N_max)
plot_N_body_problem_velocities(N_body_problem, U0_list, t, Crank_Nicolson, tol_jacobian=tol_jacobian, newton_tol=newton_tol, N_max=N_max)