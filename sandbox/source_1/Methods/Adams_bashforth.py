import numpy as np

def rk4_step(f, Y0, t, h):
    k1 = f(Y0, t) * h
    k2 = h * f(Y0 + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(Y0 + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(Y0 + k3, t + h)
    return Y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def adams_bashforth_4th(f, Y0, t):
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    h = t[1] - t[0]

    Y[1] = rk4_step(f, Y[0], t[0], h)
    Y[2] = rk4_step(f, Y[1], t[1], h)
    Y[3] = rk4_step(f, Y[2], t[2], h)

    for i in range(3, len(t) - 1):
        Y[i+1] = Y[i] + h * (55 * f(Y[i], t[i]) / 24 - 59 * f(Y[i - 1], t[i - 1]) / 24 +
                             37 * f(Y[i - 2], t[i - 2]) / 24 - 3 * f(Y[i - 3], t[i - 3]) / 8)

    return Y