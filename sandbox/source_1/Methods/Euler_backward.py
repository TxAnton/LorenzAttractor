import numpy as np
from scipy import optimize

def backward_euler_fixed_point(f, t, Y0):
    Y = np.zeros((len(t), len(Y0)))
    Y[0] = Y0
    h = t[1] - t[0]
    it_max = 10

    for i in range(len(t)-1):
        tp = t[i] + h
        yp, yp2 = Y[i], Y[i]
        for j in range(0, it_max):
            yp_save = yp
            yp = yp2 + h * f(yp, tp)
            if (np.max(np.abs(yp - yp_save)) < 1e-12):
                break

        Y[i+1] = yp
    return Y

