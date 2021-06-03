import numpy as np


def rk4(f, Y0, t):
    Y = []
    Y.append(Y0)
    h = t[1] - t[0]
    for i in range(len(t)-1):
        k1 = h * f(Y[i], t[i])
        k2 = h * f(Y[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(Y[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(Y[i] + k3, t[i] + h)
        Y.append(Y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return np.array(Y)

def rk4_step(f, Y0, t, h):
    k1 = f(Y0, t) * h
    k2 = h * f(Y0 + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(Y0 + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(Y0 + k3, t + h)
    return Y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6