import numpy as np

def tol_finder(x, x_, atol, rtol):
    tol = np.zeros(len(x))
    for i in range(len(x)):
        tol[i] = atol + max(abs(x[i]), abs(x_[i])) * rtol
    return tol

def err_finder(tol, x, x_):
    err = 0
    for i in range(len(x)):
        err += ((x_[i] - x[i]) / tol[i])**2
    return np.sqrt(err/(len(x)))

def runge_kutta_fehlberga(f, t0, Y0, tEnd, h, atol, rtol):
    Y = []
    Y.append(Y0)
    while t0 < tEnd-h:
        k1 = h * f(Y0, t0)
        k2 = h * f(Y0 + (1 / 4) * k1, t0 + (1 / 4) * h)
        k3 = h * f(Y0 + (3 / 32) * k1 + (9 / 32) * k2, t0 + (3 / 8) * h)
        k4 = h * f(Y0 + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3, t0 + (12 / 13) * h)
        k5 = h * f(Y0 + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4, t0 + h)
        k6 = h * f(Y0 - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5,
                   t0 + (1 / 2) * h)

        Y0_ = Y0 + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
        Y0 = Y0 + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6

        tol = tol_finder(Y0, Y0_, atol, rtol)
        err = err_finder(tol, Y0, Y0_)

        h = h * (1/err)**(1/(min(4, 5)+1))

        t0 = t0 + h

        Y.append(Y0)

    return np.array(Y)