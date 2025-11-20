from numpy import zeros
def Jacobian(F, x0, h=1e-5):
        #Computes the numeric Jacobian using symmetric differences. 
        #F: The function of which to compute the Jacobian.
        #x0: The point in which to compute the Jacobian.
        #h: The tolerance (scalar).
        #args: Extra arguments for f.
        
        # Compute the dimensions of the Jacobian matrix from the function and its variable.
        N = len(x0)
        
        J = zeros((N, N))
        delta_x = zeros(N)

        for i in range(0, N):
            delta_x[:] = 0
            delta_x[i] = h
            J[:, i] = (F(x0+delta_x) - F(x0-delta_x))/(2*h)

        return J