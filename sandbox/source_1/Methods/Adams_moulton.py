import numpy as np

def rk4_step(f, Y0, t, h):
    k1 = f(Y0, t) * h
    k2 = h * f(Y0 + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(Y0 + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(Y0 + k3, t + h)
    return Y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def adams_moulton_4th(f, Y0, t):
    ##PECECE
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    h = t[1] - t[0]

    Y[1] = rk4_step(f, Y[0], t[0], h)
    Y[2] = rk4_step(f, Y[1], t[1], h)
    Y[3] = rk4_step(f, Y[2], t[2], h)

    f_m2 = f(Y[0], t[0])
    f_m1 = f(Y[1], t[1])
    f_0 = f(Y[2], t[2])
    f_1 = f(Y[3], t[3])
    for i in range(3, len(t) - 1):
        f_m3, f_m2, f_m1, f_0 = f_m2, f_m1, f_0, f_1

        Y[i+1] = Y[i] + (h/24) * (55*f_0 - 59*f_m1 + 37*f_m2 - 9*f_m3)

        f_1 = f(Y[i + 1], t[i + 1])

        Y[i+1] = Y[i] + (h/24) * (9*f_1 + 19*f_0 - 5*f_m1 + f_m2)

        f_1 = f(Y[i + 1], t[i + 1])
    return Y
    return Y