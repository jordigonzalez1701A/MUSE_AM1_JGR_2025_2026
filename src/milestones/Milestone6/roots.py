def newton(func, gamma0):
        N_max = 10000
        delta = 1e-11
        gamma_n = gamma0
        gamma_n_plus_one = gamma0
        tol = 1e-10
        for n in range(0, N_max):
            f_n = func(gamma_n)
            f_prime_n = (func(gamma_n + delta) - func(gamma_n-delta)) / (2*delta)
            gamma_n_plus_one = gamma_n - f_n/f_prime_n

            if(abs(gamma_n_plus_one-gamma_n)<tol):
                return gamma_n_plus_one
            
            gamma_n = gamma_n_plus_one
        
        return gamma_n_plus_one