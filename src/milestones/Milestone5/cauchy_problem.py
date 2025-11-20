from numpy import zeros
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
