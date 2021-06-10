# логика для аттрактора Лоренца
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import animation as anim
import matplotlib.pyplot as plt
import numpy as np
import methods
import random


def progress_bar(cur, lim):
    for vv in range(9, -1, -1):
        if cur == int(lim * (float(vv) / 10.0)):
            print(vv, end="")
            if vv == 9: print()
            break


def compare(s_points, o_points, adopt_time_scale: bool = False, interp_type='linear'):  # Расстояние между двумя аттракторами
    get_distance = True

    min_t = min(s_points[3][-1], o_points[3][-1])  # Минимальное время конца
    max_t = max(s_points[3][0], o_points[3][0]) # Максимальное вермя начала
    assert (max_t < min_t)
    if adopt_time_scale:
        t = o_points[3]
    else:
        t = s_points[3]
    i = 0

    # build interpolated function for function out of time scale
    if adopt_time_scale:
        f_interp = [interp1d(s_points[3], s_points[i], kind=interp_type) for i in range(3)]
    else:
        f_interp = [interp1d(o_points[3], o_points[i], kind=interp_type) for i in range(3)]

    err = [[], [], []]  # x,y,z
    arr_interp = np.empty((len(t), 3))
    i_s = 0
    i_f = -1
    for i in range(len(t)):

        if (t[i] >= max_t):
            if (t[i] <= min_t):
                arr_interp[i] = [f_interp[j](t[i]) for j in range(3)]
            else:
                arr_interp = arr_interp[i_s:i]
                i_f = i
                break
        else: i_s += 1
    arr_interp = arr_interp.transpose()
    if adopt_time_scale:
        dts = o_points[:3,i_s:i_f]

        # err = np.abs(o_dots[:3,i_s:i_f] - arr_interp.transpose())
    else:
        dts = s_points[:3, i_s:i_f]
        # m_dim = min(dts.shape[1], arr_interp.shape[1])
        # err = np.abs(dts[:, :m_dim] - arr_interp[:, :m_dim])
        # err = np.abs(s_dots[:3,i_s:i_f] - arr_interp.transpose())
    m_dim = min(dts.shape[1], arr_interp.shape[1])
    err = np.abs(dts[:, :m_dim] - arr_interp[:, :m_dim])
    return np.vstack([np.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2), t[i_s:i_f]])


class Attractor:
    methods = ["EUL1", "MIDP2", "RK4", "AB4", "ABM5"]

    def compare(self):  # TODO remove
        pass

    """
    My class

    Bla
    """

    def __init__(self, s=10, r=28, b=2.667, step=0.01, num_steps=10000, init_value=(0., 1., 1.05), counter=0):
        self.count = 0

        self.s = s
        self.r = r
        self.b = b

        self.counter = counter
        self.step = step
        self.num_steps = num_steps
        self.init_value = init_value

        self.points = np.vstack([np.empty(self.num_steps + 1), np.empty(self.num_steps + 1),
                                 np.empty(self.num_steps + 1), np.empty(self.num_steps + 1)])
        self.points[0][0], self.points[1][0], self.points[2][0] = self.init_value

        self.func = [np.empty(self.num_steps + 1), np.empty(self.num_steps + 1), np.empty(self.num_steps + 1)]

    # CallBack для сохранения точек, если пол-ль тоже хочет сохранять все точки, реализует свой
    def savePoint(self, coord, time, index):
        self.points[0][index] = coord[0]  # x
        self.points[1][index] = coord[1]  # y
        self.points[2][index] = coord[2]  # z
        self.points[3][index] = time      # time = index * step

    def getPoints(self):
        # return np.vstack([self.points, np.arange(0, self.step * (self.num_steps + .5), self.step)])
        return self.points

    def getT(self):
        return np.arange(0,self.step*(self.num_steps+.5),self.step)

    def set_invariant_params(self, inv_num=1):
        self.inv_ready = inv_num
        if inv_num == 1:
            self.b = 2 * self.s
            # s and r arbitrary

            self.s = 10
            self.r = 28
            self.b = 2 * self.s

        elif inv_num == 2:
            self.b = 1
            self.r = 0
            # s arbitrary

            self.s = 10
            self.r = 0
            self.b = 1

        elif inv_num == 3:
            self.b = 1
            self.s = 1
            # r arbitrary

            self.s = 1
            self.r = 10
            self.b = 1

        elif inv_num == 4:
            b = 6 * self.s - 2
            self.r = 2 * self.s - 1
            # s arbitrary

            self.s = 0.50
            self.r = 2.0 * self.s - 1.0
            self.b = 6.0 * self.s - 2.0

        elif inv_num == 5:
            self.b = 4
            self.s = 1
            # r arbitrary

            self.s = 1
            self.r = 28
            self.b = 4

    def get_invariant_err(self, inv_num=1, dt=None):
        m = self.getPoints()
        s = self.s
        r = self.r
        b = self.b

        m1 = m[:, :-1]


        # m_t = m.copy()
        # m_t[3] += self.step
        # m_s = m
        # m = m_s  # ??? (1)

        if inv_num == 1:
            I = (m1[0] ** 2 - 2 * s * m1[2]) * np.exp(2 * s * m1[3])  # req b==2s

        elif inv_num == 2:
            I = (m1[1] ** 2 + m1[2] ** 2) * np.exp(2 * m1[3])  # req b=1, r=0

        elif inv_num == 3:
            I = (-r ** 2 * m1[0] ** 2 + m1[1] ** 2 + m1[2] ** 2) ** np.exp(2 * m1[3])  # req b = 1, s = 1

        elif inv_num == 4:
            I = ((((2.0 * s - 1) ** 2) * s) * m1[0] ** 2 + s * m1[1] ** 2 - (4 * s - 2) * m1[0] * m1[1] - (
                        1 / (4 * s)) * m1[
                     0] ** 4 + m1[0] ** 2 * m1[2]) * np.exp(4 * s * m1[3])  # req b = 6*s - 2, r = 2*s-1
        elif inv_num == 5:
            I = (-r * m1[0] ** 2 - m1[1] ** 2 + 2 * m1[0] * m1[1] + 0.25 * m1[0] ** 4 - m1[0] ** 2 * m1[2] + 4 * (
                        r - 1) * m1[
                     2]) * np.exp(4 * m1[3])  # req b = 4, s = 1
        err = np.empty((len(I) - 1))

        for i in range(len(err)):
            err[i] = (I[i + 1] - I[i]) / self.step

        # err = (np.abs(II) - np.abs(I)) / dt

        I = np.vstack([I, m1[3]])
        # err = np.vstack([err, m1[3]])
        err = np.vstack([err, self.getT()[:len(err)]])
        #
        return I[:len(I)], err

    def set_counter(self):
        self.counter = self.counter + 1

    def get_counter(self):
        return self.counter

    def f(self, Y, t=None):  # TODO t?
        d1 = self.s * (Y[1] - Y[0])
        d2 = self.r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - self.b * Y[2]

        self.set_counter()
        return np.vstack([d1, d2, d3])

    def call_method(self, method: str):
        if method == "EUL1":
            self.call(methods.EUL1)

        elif method == "MIDP2":
            self.call(methods.MIDP2)

        elif method == "RK4":
            self.call(methods.RK4)

        elif method == "AB4":
            self.call(methods.AB4)

        else:
            point = [self.points[0][0], self.points[1][0], self.points[2][0]]
            methods.AB4(self.step, self.num_steps, point, self.f, self.savePoint, True)


    def call(self, method):
        point = [self.points[0][0], self.points[1][0], self.points[2][0]]
        method(self.step, self.num_steps, point, self.f, self.savePoint)

    def show(self, name, str, do_show: bool = False, is_color=False):
        fig = plt.figure()
        fig.set_facecolor("mintcream")

        ax = fig.gca(projection='3d')

        if is_color:
            color = (random.random(), random.random(), random.random())
        else:
            color = "black"

        ax.plot(self.points[0], self.points[1], self.points[2], lw=0.5, color=color)

        ax.set_facecolor('mintcream')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(str)

        ax.tick_params(axis='x', colors="black")
        ax.tick_params(axis='y', colors="black")
        ax.tick_params(axis='z', colors="black")

        if do_show: plt.show()
        fig.savefig(name)

    def animation(self, method, is_color):
        self.call_method(method)
        anim.launch(np.array(list([self.getPoints()[0], self.getPoints()[1], self.getPoints()[2]])), is_color)
