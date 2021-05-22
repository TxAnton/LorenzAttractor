import numpy as np
import matplotlib.pyplot as plt
import types
import src


class AttractorLorenz:
    """
    My class

    Bla
    """

    def __init__(self, s, r, b, step, num_steps, init_value):
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

    def getX(self):
        return self.x_dots

    def getY(self):
        return self.y_dots

    def getZ(self):
        return self.z_dots

    def clearDots(self):
        self.x_dots = np.empty(self.num_steps + 1)
        self.y_dots = np.empty(self.num_steps + 1)
        self.z_dots = np.empty(self.num_steps + 1)

    def clearFunction(self):
        self.f_x = np.empty(self.num_steps + 1)
        self.f_y = np.empty(self.num_steps + 1)
        self.f_z = np.empty(self.num_steps + 1)

    def diff(self, x, y, z):
        x_dot = self.s * (y - x)
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z
        return x_dot, y_dot, z_dot

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
    def RKMethod(self, steps):
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
            self.x_dots[prev + 1] = self.x_dots[prev] + self.step * (1/6 * k_x[0] + 1/3 * k_x[1] + 1/3 * k_x[2] + 1/6 * k_x[3])
            self.y_dots[prev + 1] = self.y_dots[prev] + self.step * (1/6 * k_y[0] + 1/3 * k_y[1] + 1/3 * k_y[2] + 1/6 * k_y[3])
            self.z_dots[prev + 1] = self.z_dots[prev] + self.step * (1/6 * k_z[0] + 1/3 * k_z[1] + 1/3 * k_z[2] + 1/6 * k_z[3])

    def overclocking(self, k):
        self.f_x[0], self.f_y[0], self.f_z[0] = self.diff(self.x_dots[0], self.y_dots[0], self.z_dots[0])
        self.RKMethod(k)

        for i in range(1, k+1):
            self.f_x[i], self.f_y[i], self.f_z[i] = self.diff(self.x_dots[i], self.y_dots[i], self.z_dots[i])

        for j in range(k, self.num_steps):
            self.AdamBashfortsMethod(j)

    def AdamBashfortsMethod(self, n):
        coeff = [55/24, -59/24, 37/24, -3/8]

        temp_x = coeff[0] * self.f_x[n] + coeff[1] * self.f_x[n-1] + coeff[2] * self.f_x[n-2] + coeff[3] * self.f_x[n-3]
        self.x_dots[n + 1] = self.x_dots[n] + self.step * temp_x

        temp_y = coeff[0] * self.f_y[n] + coeff[1] * self.f_y[n - 1] + coeff[2] * self.f_y[n - 2] + coeff[3] * self.f_y[n - 3]
        self.y_dots[n + 1] = self.y_dots[n] + self.step * temp_y

        temp_z = coeff[0] * self.f_z[n] + coeff[1] * self.f_z[n - 1] + coeff[2] * self.f_z[n - 2] + coeff[3] * self.f_z[n - 3]
        self.z_dots[n + 1] = self.z_dots[n] + self.step * temp_z

        self.f_x[n + 1], self.f_y[n + 1], self.f_z[n + 1] = self.diff(self.x_dots[n + 1], self.y_dots[n + 1], self.z_dots[n + 1])

    def printDots(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        for i in range(numb):
            print("Dots[" + str(i) + "]: x = " + str(self.x_dots[i]) + "; y = " + str(self.y_dots[i]) + "; z = " + str(self.z_dots[i]))

    def printValFunc(self, numb):
        if numb > self.num_steps + 1:
            numb = self.num_steps + 1
        for i in range(numb):
            print("Value function[" + str(i) + "]: f(x) = " + str(self.f_x[i]) + "; f(y) = " + str(self.f_y[i]) + "; f(z) = " + str(self.f_z[i]))

    def createPNG(self, name):
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
        fig.savefig(name + ".png")

    # def compare(self, other:src.lorenzMethods.AttractorLorenz, adapt_time_scale:bool = False):
    #
    #     s_dots = self.getDots()
    #     o_dots = other.
