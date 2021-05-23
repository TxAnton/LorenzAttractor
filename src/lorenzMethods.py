import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class AttractorLorenz:
    """
    My class

    Bla
    """

    def __init__(self, s, r, b, step=0.01, num_steps=10000, init_value=(0., 1., 1.05)):
        self.s = s
        self.r = r
        self.b = b

        self.step = step
        self.num_steps = num_steps
        self.init_value = init_value

        self.x_dots = np.empty(self.num_steps + 1)
        self.y_dots = np.empty(self.num_steps + 1)
        self.z_dots = np.empty(self.num_steps + 1)
        self.x_dots[0], self.y_dots[0], self.z_dots[0] = self.init_value

        self.f_x = np.empty(self.num_steps + 1)
        self.f_y = np.empty(self.num_steps + 1)
        self.f_z = np.empty(self.num_steps + 1)

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

            self.s=.50
            self.r=2.0 * self.s - 1.0
            self.b=6.0*self.s - 2.0

        elif inv_num == 5:
            self.b = 4
            self.s = 1
            # r arbitrary

            self.s=1
            self.r=28
            self.b=4

    def get_invariant_err(self, inv_num=1):
        m = self.getDots()
        s = self.s
        r = self.r
        b = self.b



        # m_t = np.vstack([m[0][:-1],m[1][:-1],m[2][:-1],m[3][1:]])
        # m_s = np.vstack([m[0][:-1], m[1][:-1], m[2][:-1], m[3][:-1]])
        m_t = m.copy()
        m_t[3]+=self.step
        m_s = m


        m = m_s

        if inv_num == 1:
            I = (m[0] ** 2 - 2 * s * m[2]) * np.exp(2 * s * m[3])  # req b==2s
            II = (m_t[0] ** 2 - 2 * s * m_t[2]) * np.exp(2 * s * m_t[3])
        elif inv_num == 2:
            I = (m[1] ** 2 + m[2] ** 2) * np.exp(2 * m[3])  # req b=1, r=0
            II = (m_t[1] ** 2 + m_t[2] ** 2) * np.exp(2 * m_t[3])
        elif inv_num == 3:
            I = (-r ** 2 * m[0] ** 2 + m[1] ** 2 + m[2] ** 2) ** np.exp(2 * m[3])  # req b = 1, s = 1
            II = (-r ** 2 * m_t[0] ** 2 + m_t[1] ** 2 + m_t[2] ** 2) ** np.exp(2 * m_t[3])
        elif inv_num == 4:
            I = ((((2.0 * s - 1) ** 2) * s) * m[0] ** 2 + s * m[1] ** 2 - (4 * s - 2) * m[0] * m[1] - (1 / (4 * s)) * m[
                0] ** 4 + m[0] ** 2 * m[2]) * np.exp(4 * s * m[3])  # req b = 6*s - 2, r = 2*s-1
            II = ((((2.0 * s - 1) ** 2) * s) * m[0] ** 2 + s * m_t[1] ** 2 - (4 * s - 2) * m_t[0] * m_t[1] - (1 / (4 * s)) * m_t[
                0] ** 4 + m_t[0] ** 2 * m_t[2]) * np.exp(4 * s * m_t[3])
        elif inv_num == 5:
            I = (-r * m[0] ** 2 - m[1] ** 2 + 2 * m[0] * m[1] + 0.25 * m[0] ** 4 - m[0] ** 2 * m[2] + 4 * (r - 1) * m[
                2]) * np.exp(4 * m[3])  # req b = 4, s = 1
            II = (-r * m_t[0] ** 2 - m_t[1] ** 2 + 2 * m_t[0] * m_t[1] + 0.25 * m_t[0] ** 4 - m_t[0] ** 2 * m_t[2] + 4 * (r - 1) * m_t[
                2]) * np.exp(4 * m_t[3])

        # err = (II-I)/self.step

        err = np.empty((len(I)-1))


        for i in range(len(err)):
            err[i] = (I[i+1]-I[i])/(self.step)

        I = np.vstack([I,self.getT()])
        err = np.vstack([err, self.getT()[:len(err)]])

        return I[:len(I)], err

    def getX(self):
        return self.x_dots

    def getY(self):
        return self.y_dots

    def getZ(self):
        return self.z_dots

    def getT(self):
        return np.arange(0, self.step * (self.num_steps + .1), self.step)

    def getDots(self):
        return np.vstack([self.getX(), self.getY(), self.getZ(), self.getT()])

    def clearDots(self):
        self.x_dots = np.empty(self.num_steps + 1)
        self.y_dots = np.empty(self.num_steps + 1)
        self.z_dots = np.empty(self.num_steps + 1)

    def clearFunction(self):
        self.f_x = np.empty(self.num_steps + 1)
        self.f_y = np.empty(self.num_steps + 1)
        self.f_z = np.empty(self.num_steps + 1)

    def f(self, Y, t):  # System in standart form
        d1 = self.s * (Y[1] - Y[0])
        d2 = self.r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - self.b * Y[2]
        return np.vstack([d1, d2, d3])

    # def diff(self, x, y, z):
    #     x_dot = self.s * (y - x)
    #     y_dot = self.r * x - y - x * z
    #     z_dot = x * y - self.b * z
    #     return x_dot, y_dot, z_dot

    def __call__(self, t, Y):  # System in standart form
        s = self.s
        r = self.r
        b = self.b
        d1 = s * (Y[1] - Y[0])
        d2 = r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - b * Y[2]
        return [d1, d2, d3]

    #
    def diff(self, x, y, z):
        return self.__call__(None, np.array([x, y, z]))

    #     # return x_dot, y_dot, z_dot

    def calcDots(self, x_dot, y_dot, z_dot, prev, step):
        self.x_dots[prev + 1] = self.x_dots[prev] + step * x_dot  # x_1 = x_0 + h * f(x_0, y_0, z_0)
        self.y_dots[prev + 1] = self.y_dots[prev] + step * y_dot
        self.z_dots[prev + 1] = self.z_dots[prev] + step * z_dot

    # метод Эйлера - 1 порядок
    def EulerMethod(self):  # init other values from book!

        for prev in range(self.num_steps):
            x_dot, y_dot, z_dot = self.diff(self.x_dots[prev], self.y_dots[prev], self.z_dots[prev])
            self.calcDots(x_dot, y_dot, z_dot, prev, self.step)

    # метод средней точки - 2 порядка
    def midpointMethod(self):

        for prev in range(self.num_steps):

            x_dot, y_dot, z_dot = self.diff(self.x_dots[prev], self.y_dots[prev], self.z_dots[prev])
            self.calcDots(x_dot, y_dot, z_dot, prev, self.step / 2)

            x_dot, y_dot, z_dot = self.diff(self.x_dots[prev + 1], self.y_dots[prev + 1], self.z_dots[prev + 1])
            self.calcDots(x_dot, y_dot, z_dot, prev, self.step)

    # # метод Рунге-Кутты 4 порядка
    def RKMethod(self, steps = None):
        if steps == None: steps = self.num_steps
        k_x = np.empty(4)
        k_y = np.empty(4)
        k_z = np.empty(4)
        coeff = [2, 2, 1]
        coeff2 = [0, 1, 1]

        for prev in range(steps):
            for i in range(3):
                k_x[i], k_y[i], k_z[i] = self.diff(self.x_dots[prev + coeff2[i]], self.y_dots[prev + coeff2[i]], self.z_dots[prev + coeff2[i]])
                self.calcDots(k_x[i], k_y[i], k_z[i], prev, self.step / coeff[i])

            k_x[3], k_y[3], k_z[3] = self.diff(self.x_dots[prev + 1], self.y_dots[prev + 1], self.z_dots[prev + 1])
            self.x_dots[prev + 1] = self.x_dots[prev] + self.step * (
                    1 / 6 * k_x[0] + 1 / 3 * k_x[1] + 1 / 3 * k_x[2] + 1 / 6 * k_x[3])
            self.y_dots[prev + 1] = self.y_dots[prev] + self.step * (
                    1 / 6 * k_y[0] + 1 / 3 * k_y[1] + 1 / 3 * k_y[2] + 1 / 6 * k_y[3])
            self.z_dots[prev + 1] = self.z_dots[prev] + self.step * (
                    1 / 6 * k_z[0] + 1 / 3 * k_z[1] + 1 / 3 * k_z[2] + 1 / 6 * k_z[3])

    def overclocking(self, k, flag=False):
        self.f_x[0], self.f_y[0], self.f_z[0] = self.diff(self.x_dots[0], self.y_dots[0], self.z_dots[0])
        self.RKMethod(k)

        for i in range(1, k + 1):
            self.f_x[i], self.f_y[i], self.f_z[i] = self.diff(self.x_dots[i], self.y_dots[i], self.z_dots[i])

        for j in range(k, self.num_steps):
            self.AdamBashfortsMethod(j, flag)

    def AdamBashfortsMethod(self, n, flag=False):
        coeff = [55/24, -59/24, 37/24, -3/8]

        temp_x = coeff[0] * self.f_x[n] + coeff[1] * self.f_x[n - 1] + coeff[2] * self.f_x[n - 2] + coeff[3] * self.f_x[
            n - 3]
        self.x_dots[n + 1] = self.x_dots[n] + self.step * temp_x

        temp_y = coeff[0] * self.f_y[n] + coeff[1] * self.f_y[n - 1] + coeff[2] * self.f_y[n - 2] + coeff[3] * self.f_y[
            n - 3]
        self.y_dots[n + 1] = self.y_dots[n] + self.step * temp_y

        temp_z = coeff[0] * self.f_z[n] + coeff[1] * self.f_z[n - 1] + coeff[2] * self.f_z[n - 2] + coeff[3] * self.f_z[
            n - 3]
        self.z_dots[n + 1] = self.z_dots[n] + self.step * temp_z
        # P
        self.f_x[n + 1], self.f_y[n + 1], self.f_z[n + 1] = self.diff(self.x_dots[n + 1], self.y_dots[n + 1], self.z_dots[n + 1])
        # PE
        if flag == True:
            iter = 3
            self.AdamMoultonMethod(n, iter)

    def AdamMoultonMethod(self, n, iter):
        coeff = [251/720, 656/720, -264/720, 106/720, -19/720]

        for i in range(iter):
            # PEC # PECEC #PECECEC
            temp_x = coeff[0] * self.f_x[n + 1] + coeff[1] * self.f_x[n] + coeff[2] * self.f_x[n - 1] + coeff[3] * self.f_x[n - 2] + coeff[4] * self.f_x[n - 3]
            self.x_dots[n + 1] = self.x_dots[n] + self.step * temp_x

            temp_y = coeff[0] * self.f_y[n + 1] + coeff[1] * self.f_y[n] + coeff[2] * self.f_y[n - 1] + coeff[3] * self.f_y[n - 2] + coeff[4] * self.f_y[n - 3]
            self.y_dots[n + 1] = self.y_dots[n] + self.step * temp_y

            temp_z = coeff[0] * self.f_z[n + 1] + coeff[1] * self.f_z[n] + coeff[2] * self.f_z[n - 1] + coeff[3] * self.f_z[n - 2] + coeff[4] * self.f_z[n - 3]
            self.z_dots[n + 1] = self.z_dots[n] + self.step * temp_z

            # PECE #PECECE #PECECECE
            self.f_x[n + 1], self.f_y[n + 1], self.f_z[n + 1] = self.diff(self.x_dots[n + 1], self.y_dots[n + 1], self.z_dots[n + 1])

    def sp_ivp(self, method='DOP853'):  # решение встроенным методом
        max_t = self.step * (self.num_steps + .5)
        sol_back = solve_ivp(fun=self,
                             t_span=[0, max_t],
                             y0=self.init_value,
                             method=method,
                             t_eval=np.arange(0.0, max_t, self.step))
        y = sol_back.y
        self.x_dots = y[0]
        self.y_dots = y[1]
        self.z_dots = y[2]
#DOPRI8
        # Старая реализация

    def gen_points_optimal(self, init_vals=(0., 1., 1.05), num_steps=10000, dt=0.01, s=10, r=28, b=2.667):
        # Need one more for the initial values
        init_vals = self.init_value
        num_steps = self.num_steps
        dt = self.step
        s = self.s

        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)

        ts = np.empty(num_steps + 1)
        # Set initial values
        xs[0], ys[0], zs[0] = init_vals
        ts[0] = 0

        lorenz = lambda x, y, z, s, r, b: self.diff(x, y, z)

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            lorenz(xs[i], ys[i], zs[i], s, r, b)
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
            xs[i + 1] = xs[i] + (dt / 2) * (x_dot)  # x_1=x_0+h*f(x_0, y_0, z_0)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
            x_dot2, y_dot2, z_dot2 = lorenz(xs[i + 1], ys[i + 1], zs[i + 1], s, r, b)
            xs[i + 1] = xs[i] + (dt / 2) * (x_dot + x_dot2)
            ys[i + 1] = ys[i] + (dt / 2) * (y_dot + y_dot2)
            zs[i + 1] = zs[i] + (dt / 2) * (z_dot + z_dot2)
            ts[i + 1] = ts[i] + dt

        self.x_dots = xs
        self.y_dots = ys
        self.z_dots = zs
        # dots = [xs, ys, zs, ts]
        # return dots

    def printDots(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        for i in range(numb):
            print("Dots[" + str(i) + "]: x = " + str(self.x_dots[i]) + "; y = " + str(self.y_dots[i]) + "; z = " + str(
                self.z_dots[i]))

    def printValFunc(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        for i in range(numb):
            print("Value function[" + str(i) + "]: f(x) = " + str(self.f_x[i]) + "; f(y) = " + str(
                self.f_y[i]) + "; f(z) = " + str(self.f_z[i]))

    def show(self):
        fig = plt.figure()
        fig.set_facecolor("mintcream")

        ax = fig.gca(projection='3d')
        ax.plot(self.x_dots, self.y_dots, self.z_dots, lw=0.5)

        ax.set_facecolor('mintcream')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        ax.tick_params(axis='x', colors="orange")
        ax.tick_params(axis='y', colors="orange")
        ax.tick_params(axis='z', colors="orange")

        plt.show()

    def createPNG(self, name, do_show: bool = True):
        fig = plt.figure()
        fig.set_facecolor("mintcream")

        ax = fig.gca(projection='3d')
        ax.plot(self.x_dots, self.y_dots, self.z_dots, lw=0.5)

        ax.set_facecolor('mintcream')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        ax.tick_params(axis='x', colors="orange")
        ax.tick_params(axis='y', colors="orange")
        ax.tick_params(axis='z', colors="orange")

        if do_show: plt.show()
        fig.savefig(name.rstrip(".png") + ".png")

    def compare(self, other, get_distance=True, adopt_time_scale: bool = False,
                interp_type='linear'):  # Расстояние между двумя аттракторами

        s_dots = self.getDots()
        o_dots = other.getDots()

        min_t = min(s_dots[3][-1], o_dots[3][-1])

        if adopt_time_scale:
            t = o_dots[3]
        else:
            t = s_dots[3]
        i = 0

        # build interpolated function for function out of time scale
        if adopt_time_scale:
            f_interp = [interp1d(s_dots[3], s_dots[i], kind=interp_type) for i in range(3)]
        else:
            f_interp = [interp1d(o_dots[3], o_dots[i], kind=interp_type) for i in range(3)]

        err = [[], [], []]  # x,y,z
        arr_interp = np.empty((len(t), 3))
        for i in range(len(t)):
            if (t[i] <= min_t):
                arr_interp[i] = [f_interp[j](t[i]) for j in range(3)]
            else:
                arr_interp = arr_interp[:i]
                break

        if adopt_time_scale:
            err = np.abs(o_dots[:3] - arr_interp.transpose())
        else:
            err = np.abs(s_dots[:3] - arr_interp.transpose())
        if (get_distance):
            return np.vstack([np.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2), t])
        else:
            return np.vstack([err, t])
