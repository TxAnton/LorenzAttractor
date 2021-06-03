from Methods.RK4 import rk4_step
import numpy as np

class Ode_solver(object):

    def __init__(self, f, method):
        self.f = f
        self.method = method
        self.y_pred = None
        self.h = None
        self.t = None
        self.atol = None
        self.rtol = None
        self.t_arr = None
        self.f_arr = None
        self.y_arr = None


    def tol_finder(self, x_):
        tol = np.zeros(len(self.y_pred))
        for i in range(len(self.y_pred)):
            tol[i] = self.atol + max(abs(self.y_pred[i]), abs(x_[i])) * self.rtol
        return tol


    def err_finder(self, tol, x_):
        err = 0
        for i in range(len(self.y_pred)):
            err += ((x_[i] - self.y_pred[i]) / tol[i]) ** 2
        return np.sqrt(err / (len(self.y_pred)))


    def set_params(self, t, Y0, atol=0.000000000001, rtol=0.000000000001,  y_pred = None, h=0.1):
        self.t = t
        self.atol = atol
        self.rtol = rtol
        self.h = h
        self.y_pred = Y0
        if(y_pred != None):
            self.y_pred = y_pred

        if(self.method == "AB"):
            self.y_arr = np.zeros((4, len(Y0)))
            self.t_arr = np.arange(t, h * 4 + t, h)
            self.t = self.t_arr[3]
            self.y_arr[0] = Y0
            self.y_arr[1] = rk4_step(self.f, self.y_arr[0], self.t_arr[0], h)
            self.y_arr[2] = rk4_step(self.f, self.y_arr[1], self.t_arr[1], h)
            self.y_arr[3] = rk4_step(self.f, self.y_arr[2], self.t_arr[2], h)

        if(self.method == "AM"):
            self.y_arr = np.zeros((4, len(Y0)))
            self.t_arr = np.arange(t, h * 4 + t, h)
            self.t = self.t_arr[3]
            self.y_arr[0] = Y0
            self.y_arr[0] = Y0
            self.y_arr[1] = rk4_step(self.f, self.y_arr[0], self.t_arr[0], h)
            self.y_arr[2] = rk4_step(self.f, self.y_arr[1], self.t_arr[1], h)
            self.y_arr[3] = rk4_step(self.f, self.y_arr[2], self.t_arr[2], h)
            self.f_arr = np.zeros((5, len(Y0)))
            self.f_arr[3] = self.f(self.y_arr[0], self.t_arr[0])
            self.f_arr[2] = self.f(self.y_arr[1], self.t_arr[1])
            self.f_arr[1] = self.f(self.y_arr[2], self.t_arr[2])
            self.f_arr[0] = self.f(self.y_arr[3], self.t_arr[3])

    def integrate(self, new_t=None):
        if(self.method == "EB"): ##backward Euler
            it_max = 10
            self.h = np.abs(new_t - self.t)
            tp = new_t
            yp, yp2 = self.y_pred, self.y_pred
            for j in range(0, it_max):
                yp_save = yp
                yp = yp2 + self.h * self.f(yp, tp)
                if (np.max(np.abs(yp - yp_save)) < 1e-12):
                    break
            self.y_pred = yp
            self.t = new_t
            return yp

        if(self.method == "RK4"):
            h = np.abs(new_t - self.t)
            k1 = self.f(self.y_pred, new_t) * h
            k2 = h * self.f(self.y_pred + 0.5 * k1, new_t + 0.5 * h)
            k3 = h * self.f(self.y_pred + 0.5 * k2, new_t + 0.5 * h)
            k4 = h * self.f(self.y_pred + k3, new_t + h)
            self.t = new_t
            Y = self.y_pred + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.y_pred = Y
            return Y

        if(self.method == "RK45"):
            k1 = self.h * self.f(self.y_pred, self.t)
            k2 = self.h * self.f(self.y_pred + (1 / 4) * k1, self.t + (1 / 4) * self.h)
            k3 = self.h * self.f(self.y_pred + (3 / 32) * k1 + (9 / 32) * k2, self.t + (3 / 8) * self.h)
            k4 = self.h * self.f(self.y_pred + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3, self.t + (12 / 13) * self.h)
            k5 = self.h * self.f(self.y_pred + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4, self.t + self.h)
            k6 = self.h * self.f(self.y_pred - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5,
                       self.t + (1 / 2) * self.h)

            Y0_ = self.y_pred + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
            self.y_pred = self.y_pred + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6

            tol = self.tol_finder(Y0_)
            err = self.err_finder(tol, Y0_)

            self.h = self.h * (1 / err) ** (1 / (min(4, 5) + 1))
            self.t += self.h
            return self.y_pred

        if(self.method == "AB"):
            self.h = self.t_arr[1] - self.t_arr[0]
            self.t = new_t
            self.y_pred = self.y_arr[3] + self.h * (55 * self.f(self.y_arr[3], 0) / 24 -
                                                    59 * self.f(self.y_arr[2], 0) / 24 +
                                                    37 * self.f(self.y_arr[1], 0) / 24 -
                                                    3 * self.f(self.y_arr[0], 0) / 8)
            self.t_arr = np.roll(self.t_arr, -1)
            self.t_arr[3] = new_t
            self.y_arr = np.roll(self.y_arr, -1, axis=0)
            self.y_arr[3] = self.y_pred
            return self.y_pred

        if(self.method == "AM"):
            h = self.t_arr[1] - self.t_arr[0]
            self.t = new_t
            self.f_arr[4], self.f_arr[3], self.f_arr[2], self.f_arr[1] = self.f_arr[3], self.f_arr[2], self.f_arr[1], self.f_arr[0]

            Y = self.y_pred + (h / 24) * (55 * self.f_arr[0] - 59 * self.f_arr[1] +
                                          37 * self.f_arr[2] - 9 * self.f_arr[3])
            self.f_arr[0] = self.f(Y, new_t)

            Y = self.y_pred + (h / 24) * (9 * self.f_arr[0] + 19 * self.f_arr[1] - 5 * self.f_arr[2] + self.f_arr[3])
            self.f_arr[0] = self.f(Y, new_t)
            self.y_pred = Y
            return Y