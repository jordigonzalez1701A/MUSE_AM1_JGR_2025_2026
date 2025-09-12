# JORDI GONZÁLEZ DE REGÀS -- Ampliación de Matemáticas 1, Máster Universitario en Sistemas Espaciales (MUSE) UPM, curso 2025-26.
# This script uses Euler's method to solve a Kepler's initial value problem (IVP):
# r'' = - r / |r|^3, 
# r(0) = (1,0)
# r'(0) = (0,1)
# where r is the position vector, |r| is its modulus, and r'' is its second time derivative. 
# A 4-vector U is defined as the coordinates in the phase space U=(x,y,x',y'), so the equation becomes U' = F, where
# F = (r, -r/|r|^3)


from numpy import array, concatenate, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

def F(U):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    """
    r = U[0:2]
    rd = U[2:4]

    return concatenate((rd, -r/norm(r)**3), axis=None)

N = 200 # Maximum order of Euler's method.
delta_t = 1 
U = array([1,0,0,1])

x = zeros(N)
y = zeros(N)

x[0] = U[0]
y[0] = U[1]

for i in range(1, N):
    # Compute each step of Euler's method
    F_value = F(U)
    U = U + delta_t * F_value

    x[i] = U[0]
    y[i] = U[1]

plt.plot(x, y)
plt.show()