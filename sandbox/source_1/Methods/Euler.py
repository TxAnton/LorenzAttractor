import numpy as np


def euler(f, Y0, tau, t):
    Y = []
    Y.append(Y0)
    for n in range(len(t) - 1):
        Y.append(Y[n] + tau * f(Y[n], t[n]))
    return np.array(Y)

def euler_step(f, tau, t, Y):
    return Y + tau * f(Y, t)


def euler_test_cycle(f, Y0, t, deltaT):
    for i in range(100000):
        sol = []
        sol.append(Y0)
        for n in range(len(t) - 1):
            sol.append(euler_step(f, deltaT, t[n], sol[n]))
        sol = np.array(sol)


def euler_test_func(f, Y0, t, deltaT):
    for i in range(100000):
        euler(f, Y0, deltaT, t)