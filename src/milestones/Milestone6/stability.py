from numpy import zeros
from numpy.linalg import eigvals
def Jacobian(f, U, t, delta=1e-8):
    """
    Returns the Jacobian of a function of signature f(U,t,mu)
    """
    N =len(U)
    J = zeros((N,N))
    for i in range(0, N):
        h = zeros(N)
        h[i] = delta
        J[:,i] = (f(U+h,t)-f(U-h,t))/(2*delta)

    return J

def stability_point(f, U, t, delta=1e-8):

    J = Jacobian(f, U, t, delta=1e-8)
    return eigvals(J)
