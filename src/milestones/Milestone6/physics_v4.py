from numpy import concatenate, array, zeros, abs, sqrt, reshape
from numpy.linalg import norm

def Kepler(U, t):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    Note: F is not explicitly dependent on t in this case, adding it allows to construct
    robust functions for the numerical methods.
    """
    r = U[0:2]
    rd = U[2:4]

    return concatenate((rd, -r/norm(r)**3), axis=None)

def linear_oscillator(U, t):
    """
    Returns F(U,t)=(y,-x)
    """
    return array([U[1], -U[0]])

def N_body_problem(U, t, N_body=5):

    """
    Returns F(U)=(v0, sum_0, v1, sum_1, ...) for the N-body problem, where sum = \sum_{j=0,i\neq j}^{N_body-1}r_j-r_i/||r_j-r_i||**3.
    INPUTS:
    U: State vector (r0, v0, r1, v1, r2, v2,..., r_(Nb-1), v_(Nb-1))
    t: time.
    """
    dim = 3
    F = zeros(2*N_body*dim)
    
    for i in range(0, N_body):
        # Carga de velocidades
        
        F[2*i*dim:(2*i+1)*dim] = U[(2*i+1)*dim:(2*i+2)*dim]
            
        sum = zeros(dim)
        for j in range(0, N_body):
            if j != i:
                sum = sum + (U[2*j*dim:(2*j+1)*dim]-U[2*i*dim:(2*i+1)*dim])/norm(U[2*j*dim:(2*j+1)*dim]-U[2*i*dim:(2*i+1)*dim])**3
        
        F[(2*i+1)*dim:(2*i+2)*dim] = sum[:]
    
    return F

def CRTBP(U, t):
    """
    Calcula F(U, mu, t), donde U es el vector 
    estado.

    Inputs:
    U: Vector estado del sistema. U=(x,y,z,vx,vy,vz)
    mu: Masa reducida.
    t: tiempo adimensional.

    """
    
    with open("mu.txt", "r") as f:
        mu = float(f.read().strip())
    # CRTBP
    d1 = sqrt((U[0]+mu)**2 + U[1]**2 + U[2]**2)
    d2 = sqrt((U[0]-1+mu)**2 + U[1]**2 + U[2]**2)
    G = zeros(6)
    G[0:3] = U[3:6]
    G[3] = U[0] + 2*U[4]-(1-mu)*(U[0]+mu)/d1**3-mu*(U[0]-1+mu)/d2**3
    G[4] = -2*U[3]+U[1]-((1-mu)/d1**3 + mu/d2**3)*U[1]
    G[5] = -((1-mu)/d1**3 + mu/d2**3)*U[2]

    return G