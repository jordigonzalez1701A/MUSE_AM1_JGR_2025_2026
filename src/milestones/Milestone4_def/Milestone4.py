from numpy import sqrt, real, imag
from numpy.random import uniform
from numpy import meshgrid
import matplotlib.pyplot as plt
from temporal_schemes_v2 import Cauchy_problem, Euler, Crank_Nicolson, RK4, Inverse_Euler, leapfrog
from differential_equation_v2 import linear_oscillator

def stability_region(temporal_scheme):
    """
    Finds and plots the region of absolute stability of a given temporal scheme.
    r: U^(n+1)
    omega: delta_t
    Inputs:
    temporal_scheme: the temporal scheme.
    """
    if temporal_scheme == Euler:
        def r(omega):
            return 1+omega
    elif temporal_scheme == Crank_Nicolson:
        def r(omega):
            return (1+0.5*omega)/(1-0.5*omega)
    elif temporal_scheme == Inverse_Euler:
        def r(omega):
           return  1/(1-omega)
    elif temporal_scheme == RK4:
        def r(omega):
            return 1+omega+0.5*omega**2+(1/6)*omega**3+(1/24)*omega**4
    elif temporal_scheme == leapfrog:
        def r_one(omega):
            #https://www.sciencedirect.com/science/article/pii/S0096300308008758
            
            return omega + sqrt(omega**2 + 1)
        def r_two(omega):
            #https://www.sciencedirect.com/science/article/pii/S0096300308008758
            
            return omega - sqrt(omega**2 + 1)    
    else:
        print("ERROR: Numerical method not available")
        return
    N = 500
    real_part = uniform(-5, 5, N)
    imag_part = uniform(-5, 5, N)
    if temporal_scheme == leapfrog:
        real_part = 0
        imag_part = uniform(-5, 5, N)
    Re, Im = meshgrid(real_part, imag_part)
    omega = Re + 1j*Im
    if temporal_scheme != leapfrog:
        if temporal_scheme == Inverse_Euler or temporal_scheme == Crank_Nicolson or temporal_scheme == RK4:
            plt.scatter(real(omega[abs(r(omega))<1]), imag(omega[abs(r(omega))<1]))
        else:
            plt.contourf(real(omega), imag(omega), abs(r(omega)), levels=[0,1], colors=['#1f77b4'], alpha=0.9)
    else:
        mask = (abs(r_one(omega)) < 1) & (abs(r_two(omega)) < 1)
        print(omega[mask])
        plt.plot(real(omega[mask]), imag(omega[mask]))
    #plt.scatter(real(omega_region), imag(omega_region))
    
    plt.axis("equal")
    plt.axhline(0, color='black', linewidth=1)  # horizontal axis (y=0)
    plt.axvline(0, color='black', linewidth=1)  # vertical axis (x=0)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel(r"Re($\omega$)")
    plt.ylabel(r"Im($\omega$)")
    plt.show()

stability_region(RK4)

