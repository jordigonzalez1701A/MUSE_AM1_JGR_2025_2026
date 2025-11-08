from numpy import concatenate
from numpy.linalg import norm

def F(U, t):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    Note: F is not explicitly dependent on t in this case, adding it allows to construct
    robust functions for the numerical methods.
    """
    r = U[0:2]
    rd = U[2:4]

    return concatenate((rd, -r/norm(r)**3), axis=None)
