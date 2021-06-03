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
        self.count = 0

        self.s = s
        self.r = r
        self.b = b

        self.step = step
        self.num_steps = num_steps
        self.init_value = init_value

        x_dots = np.empty(self.num_steps + 1)
        y_dots = np.empty(self.num_steps + 1)
        z_dots = np.empty(self.num_steps + 1)

        self.dots = [x_dots, y_dots, z_dots]
        self.dots[0][0], self.dots[1][0], self.dots[2][0] = self.init_value

        f_x = np.empty(self.num_steps + 1)
        f_y = np.empty(self.num_steps + 1)
        f_z = np.empty(self.num_steps + 1)

        self.func = [f_x, f_y, f_z]

    def diff(self, x, y, z):
        x_dot = self.s * (y - x)
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z
        return x_dot, y_dot, z_dot

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
            self.b = 6.0*self.s - 2.0

        elif inv_num == 5:
            self.b = 4
            self.s = 1
            # r arbitrary

            self.s = 1
            self.r = 28
            self.b = 4

    def get_invariant_err(self, inv_num=1):
        m = self.getDots()
        s = self.s
        r = self.r
        b = self.b

        m_t = m.copy()
        m_t[3] += self.step
        m_s = m
        m = m_s  # ??? (1)

        if inv_num == 1:
            I = (m[0] ** 2 - 2 * s * m[2]) * np.exp(2 * s * m[3])  # req b==2s
        elif inv_num == 2:
            I = (m[1] ** 2 + m[2] ** 2) * np.exp(2 * m[3])  # req b=1, r=0
        elif inv_num == 3:
            I = (-r ** 2 * m[0] ** 2 + m[1] ** 2 + m[2] ** 2) ** np.exp(2 * m[3])  # req b = 1, s = 1
        elif inv_num == 4:
            I = ((((2.0 * s - 1) ** 2) * s) * m[0] ** 2 + s * m[1] ** 2 - (4 * s - 2) * m[0] * m[1] - (1 / (4 * s)) * m[
                0] ** 4 + m[0] ** 2 * m[2]) * np.exp(4 * s * m[3])  # req b = 6*s - 2, r = 2*s-1
        elif inv_num == 5:
            I = (-r * m[0] ** 2 - m[1] ** 2 + 2 * m[0] * m[1] + 0.25 * m[0] ** 4 - m[0] ** 2 * m[2] + 4 * (r - 1) * m[
                2]) * np.exp(4 * m[3])  # req b = 4, s = 1
        err = np.empty((len(I)-1))


        for i in range(len(err)):
            err[i] = (I[i + 1] - I[i]) / self.step

        I = np.vstack([I, self.getT()])
        err = np.vstack([err, self.getT()[:len(err)]])

        return I[:len(I)], err


    def clearDots(self):
        self.dots[0] = np.empty(self.num_steps + 1)
        self.dots[1] = np.empty(self.num_steps + 1)
        self.dots[2] = np.empty(self.num_steps + 1)

    def clearFunction(self):
        self.func[0] = np.empty(self.num_steps + 1)
        self.func[1] = np.empty(self.num_steps + 1)
        self.func[2] = np.empty(self.num_steps + 1)

    def f(self, Y, t):  # System in standart form
        d1 = self.s * (Y[1] - Y[0])
        d2 = self.r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - self.b * Y[2]
        return np.vstack([d1, d2, d3])

    def __call__(self, t, Y):  # System in standart form
        self.count = self.count + 1

        s = self.s
        r = self.r
        b = self.b
        d1 = s * (Y[1] - Y[0])
        d2 = r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - b * Y[2]
        return [d1, d2, d3]

    def diff(self, x, y, z):
        return self.__call__(None, np.array([x, y, z]))
    # # нужно потом сделать так, чтобы можно было подавать не только систему Лоренца (2)
    # # проверить что ничего не испортилось (в плане точности) (4)
    def counter(self):
        print("Количество вызовов функции: ", self.count)
        self.count = 0

    def iteration(self, method): # тут тоже надо как-то упростить (3)
        print("Count = ", self.count)
        print()  #
        self.printValFunc(5)            #

        if method == "EUL1":
            for index in range(self.num_steps):
                self.EulerMethod(index)

                print()                 #
                print("Count = ", self.count)
                print()                 #
                self.printValFunc(5)    #

            # сейчас сохраняет в поля класса, но вообще должен возвращать по 1 точке и сохранять куда-то
            # что-то[index+1] =

        elif method == "MIDP2":
            for index in range(self.num_steps):
                self.midpointMethod(index)

        elif method == "RK4":
            for index in range(self.num_steps):
                self.RKMethod(index)

        elif method == "ABSFRT" or method == "AMLTN" or method == "ABSFRT_MLTN":
            # разгон
            for i in range(3):
                self.RKMethod(i)
            for j in range(4):
                self.func[0][j], self.func[1][j], self.func[2][j] = self.diff(self.dots[0][j], self.dots[1][j], self.dots[2][j])
            if method == "ABSFRT" or method == "ABSFRT_MLTN":
                for index in range(3, self.num_steps):
                    self.AdamBashfortsMethod(index)
                    if method == "ABSFRT_MLTN":
                        count_PECEC = 3
                        self.AdamMoultonMethod(index, count_PECEC)
            elif method == "AMLTN":
                for index in range(3, self.num_steps + 1):
                    count_PECEC = 3
                    self.AdamMoultonMethodOnlyRK(index, count_PECEC)

    def calcDots(self, dot, index, step):
        for i in range(len(self.dots)):
            self.dots[i][index + 1] = self.dots[i][index] + step * dot[i]  # x_1 = x_0 + h * f(x_0, y_0, z_0)

    # метод Эйлера - 1 порядка
    def EulerMethod(self, index):

        x_dot, y_dot, z_dot = self.diff(self.dots[0][index], self.dots[1][index], self.dots[2][index])
        self.calcDots([x_dot, y_dot, z_dot], index, self.step)

    # метод средней точки - 2 порядка
    def midpointMethod(self, index):

        x_dot, y_dot, z_dot = self.diff(self.dots[0][index], self.dots[1][index], self.dots[2][index])
        self.calcDots([x_dot, y_dot, z_dot], index, self.step / 2)

        x_dot, y_dot, z_dot = self.diff(self.dots[0][index + 1], self.dots[1][index + 1], self.dots[2][index + 1])
        self.calcDots([x_dot, y_dot, z_dot], index, self.step)

    # # метод Рунге-Кутты 4 порядка
    def RKMethod(self, index):

        k = [np.empty(4), np.empty(4), np.empty(4)]
        coeff = [2, 2, 1]
        coeff2 = [0, 1, 1]

        for i in range(3):
            k[0][i], k[1][i], k[2][i] = self.diff(self.dots[0][index + coeff2[i]],
                                                  self.dots[1][index + coeff2[i]],
                                                  self.dots[2][index + coeff2[i]])

            self.calcDots([k[0][i], k[1][i], k[2][i]], index, self.step / coeff[i])

        k[0][3], k[1][3], k[2][3] = self.diff(self.dots[0][index + 1], self.dots[1][index + 1], self.dots[2][index + 1])
        for i in range(len(self.dots)):
            temp = (1 / 6 * k[i][0] + 1 / 3 * k[i][1] + 1 / 3 * k[i][2] + 1 / 6 * k[i][3])
            self.dots[i][index + 1] = self.dots[i][index] + self.step * temp

# вообще должна будет возвращать только значение точки) (4)
    def AdamBashfortsMethod(self, index):
        coeff = [55 / 24, -59 / 24, 37 / 24, -3 / 8]
        for i in range(len(self.dots)):
            temp = coeff[0] * self.func[i][index] + coeff[1] * self.func[i][index - 1] + coeff[2] * self.func[i][index - 2] + coeff[3] * self.func[i][index - 3]
            self.dots[i][index + 1] = self.dots[i][index] + self.step * temp
        self.func[0][index + 1], self.func[1][index + 1], self.func[2][index + 1] = self.diff(self.dots[0][index + 1],
                                                                                              self.dots[1][index + 1],
                                                                                              self.dots[2][index + 1])
        # if flag == True:
        #     iter = 3
        #     self.AdamMoultonMethod(n, iter)

    def AdamMoultonMethod(self, index, iter):
        coeff = [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]

        for j in range(iter):
            # PEC # PECEC #PECECEC
            for i in range(len(self.dots)):
                temp = coeff[0] * self.func[i][index + 1] + coeff[1] * self.func[i][index] + coeff[2] * self.func[i][index - 1] + coeff[3] * \
                        self.func[i][index - 2] + coeff[4] * self.func[i][index - 3]
                self.dots[i][index + 1] = self.dots[i][index] + self.step * temp

            # PECE #PECECE #PECECECE
            self.func[0][index + 1], self.func[1][index + 1], self.func[2][index + 1] = self.diff(self.dots[0][index + 1], self.dots[1][index + 1], self.dots[2][index + 1])
    #
    def AdamMoultonMethodOnlyRK(self, index, iter):
        coeff = [3 / 8, 19 / 24, -5 / 24, 1 / 24]

        for j in range(iter):
            for i in range(len(self.dots)):
                temp = coeff[0] * self.func[i][index] + coeff[1] * self.func[i][index - 1] + coeff[2] * self.func[i][index - 2] + coeff[3] * self.func[i][index - 3]
                self.dots[i][index] = self.dots[i][index - 1] + self.step * temp

            self.func[0][index], self.func[1][index], self.func[2][index] = self.diff(self.dots[0][index], self.dots[1][index], self.dots[2][index])

    def sp_ivp(self, method='DOP853'):  # решение встроенным методом
        max_t = self.step * (self.num_steps + .5)
        sol_back = solve_ivp(fun=self,
                             t_span=[0, max_t],
                             y0=self.init_value,
                             method=method,
                             t_eval=np.arange(0.0, max_t, self.step))
        y = sol_back.y
        self.dots[0] = y[0]
        self.dots[1] = y[1]
        self.dots[2] = y[2]
    # DOPRI8
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
        dots = [xs, ys, zs, ts]
        return dots

    def printDots(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        for i in range(numb):
            print("Dots[" + str(i) + "]: x = " + str(self.dots[0][i]) + "; y = " + str(self.dots[1][i]) + "; z = " + str(self.dots[2][i]))

    def printValFunc(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        # for i in range(numb):
        #     print("Value function[" + str(i) + "]: f(x) = " + str(self.func[0][i]) + "; f(y) = " + str(
        #         self.func[1][i]) + "; f(z) = " + str(self.func[2][i]))

        print(self.dots[0][9999], self.dots[1][9999], self.dots[2][9999])

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

    def createPNG(self, name, str, do_show: bool = True):
        fig = plt.figure()
        fig.set_facecolor("mintcream")

        ax = fig.gca(projection='3d')
        ax.plot(self.dots[0], self.dots[1], self.dots[2], lw=0.5)

        ax.set_facecolor('mintcream')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(str)

        ax.tick_params(axis='x', colors="orange")
        ax.tick_params(axis='y', colors="orange")
        ax.tick_params(axis='z', colors="orange")

        if do_show: plt.show()
        fig.savefig(name.rstrip(".png") + ".png")

