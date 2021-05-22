import random
import ffmpeg
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3

from matplotlib import animation
from src.lorenzMethods import AttractorLorenz
from matplotlib import pyplot as plt

class AnimatedAL:
    def __init__(self, N, data):
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.N = N
        self.data = data
        self.line, = self.ax.plot(self.data[0, 0:1], self.data[1, 0:1], self.data[2, 0:1], color='red')

    def settingsFig(self):
        self.fig.set_facecolor("mintcream")

    def settingsAX(self):
        self.ax.set_facecolor('mintcream')
        self.ax.tick_params(axis='x', colors="orange")
        self.ax.tick_params(axis='y', colors="orange")
        self.ax.tick_params(axis='z', colors="orange")

        self.ax.set_xlim3d([-20.0, 20.0])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-20.0, 30.0])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([10.0, 50.0])
        self.ax.set_zlabel('Z')

    def update(self, num):
        self.line.set_data(self.data[:2, :num])
        self.line.set_3d_properties(self.data[2, :num])
        color = (random.random(), random.random(), random.random())
        self.line.set_color(color)

    def play(self):
        self.settingsFig()
        self.settingsAX()
        anim = animation.FuncAnimation(self.fig, self.update, self.N, interval=10000 / self.N, blit=False)
        # anim.save("Lorenz_Attractor.git", writer='imagemagick')
        plt.show()

AL3 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
AL3.RKMethod()
AL_anim = AnimatedAL(10000, np.array(list([AL3.x_dots, AL3.y_dots, AL3.z_dots])))
AL_anim.play()


# def update(num, data, line):
#     line.set_data(data[:2, :num])
#     line.set_3d_properties(data[2, :num])
#     color = (random.random(), random.random(), random.random())
#     line.set_color(color)
#
# fig = plt.figure()
# fig.set_facecolor("mintcream")
#
# ax = p3.Axes3D(fig)
# ax.set_facecolor('mintcream')
# ax.tick_params(axis='x', colors="orange")
# ax.tick_params(axis='y', colors="orange")
# ax.tick_params(axis='z', colors="orange")
#
# N = 10000
#
# # RK4
# AL3 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
# AL3.RKMethod()
#
# data = np.array(list([AL3.getX(), AL3.getY(), AL3.getZ()]))
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], color='red')
#
# # Setting the axes properties
# ax.set_xlim3d([-20.0, 20.0])
# ax.set_xlabel('X')
#
# ax.set_ylim3d([-20.0, 30.0])
# ax.set_ylabel('Y')
#
# ax.set_zlim3d([10.0, 50.0])
# ax.set_zlabel('Z')
#
# anim = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
# #anim.save("Lorenz_Attractor.git", writer='imagemagick')
# plt.show()