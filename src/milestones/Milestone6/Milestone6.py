from numpy import array, concatenate, zeros, linspace, dot, sum, sqrt
from numpy.linalg import norm, solve, LinAlgError
from numpy import sum as npsum
from matplotlib import pyplot as plt
def single_ERK_step(F, U, t1, t2, e=None, butcher_array=None, b=None, c=None, reuse_k = False, k=None, **kwargs):
    h = t2-t1
    if reuse_k == False:
        ks = zeros((e, len(U)))
        U_stage = U.copy()
        for i in range(0, e):
            for j in range(0,i):
                U_stage += h*butcher_array[i,j]*ks[j]
                print(f"i: {i}, j: {j}, U_stage: {U_stage}, len(U_stage):{len(U_stage)}")
            
            k_i = F(U_stage, t1 + c[i]*h, **kwargs)
            print(f"i: {i}, k_i: {k_i}")
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


def CRTBP(U, t):
    """
    Calcula F(U, mu, t), donde U es el vector 
    estado.

    Inputs:
    U: Vector estado del sistema. U=(x,y,z,vx,vy,vz)
    mu: Masa reducida.
    t: tiempo adimensional.

    """
    mu = 3.004e-6

    # CRTBP
    d1 = sqrt((U[0]+mu)**2 + U[1]**2 + U[2]**2)
    d2 = sqrt((U[0]-1+mu)**2 + U[1]**2 + U[2]**2)
    G = zeros(6)
    G[0:3] = U[3:6]
    G[3] = U[0] + 2*U[4]-(1-mu)*(U[0]+mu)/d1**3-mu*(U[0]-1+mu)/d2**3
    G[4] = -2*U[3]+U[1]-((1-mu)/d1**3 + mu/d2**3)*U[1]
    G[5] = -((1-mu)/d1**3 + mu/d2**3)*U[2]

    return G
    

def plot_CRTBP_orbits(problem, U0_list, t, time_scheme, N_body=2, **kwargs):
    """
    Plots orbits with multiple initial conditions on a single plot.
    problem: The cauchy problem to solve.
    U0_list: List of initial conditions
    t: Time values for the integration (array).
    time_scheme: Time scheme for integration.
    """
    ax = plt.figure().add_subplot(projection='3d')
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)     
        ax.plot(U[:,0], U[:,1], U[:,2])
        
    ax.set_title(f"Posiciones del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
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
    for U0 in U0_list:
        U = Cauchy_problem(problem, U0, t, time_scheme, **kwargs)
        
        ax.plot(U[:,3], U[:,4], U[:,5])
        
    ax.set_title(f"Velocidades del problema de los {N_body} cuerpos con esquema {time_scheme.__name__}")
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_zlabel("v_z")
    plt.show()
    return      
    
U0 = array([1.1485693068245870E+0,-2.7775052552740525E-29,4.2187022883073371E-34,-5.9957947040197223E-16,3.7685701708850466E-2,-5.0447001281110806E-34])
U0_list = [U0]

tol_jacobian = 1e-9
N_max = 10000
newton_tol = 1e-10          

e, a, b1, b2, c, q = Butcher_tableau("Dormand-Prince")
t = linspace(0, 10, 1000)
#U = Cauchy_problem(CRTBP, U0, t, embedded_RK, e=e, butcher_array=a, b_1=b1, b_2=b2, c=c, q=q, ERK_tol=1e-6)

plot_CRTBP_velocities(CRTBP, U0_list, t, embedded_RK, N_body=2, e=e, butcher_array=a, b_1=b1, b_2=b2, c=c, q=q, ERK_tol=1e-6)