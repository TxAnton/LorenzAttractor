import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from matplotlib import animation
import random

def update(num, data, line, is_color=False):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

    if is_color:
        color = ['darkgreen', 'green', 'g']
        line.set_color(color[random.randint(0, 2)])
    else:
        line.set_color("black")


def properties(ax):
    ax.set_facecolor('mintcream')

    ax.tick_params(axis='x', colors="black")
    ax.tick_params(axis='y', colors="black")
    ax.tick_params(axis='z', colors="black")

    # Setting the axes properties
    ax.set_xlim3d([-20.0, 20.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-20.0, 30.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([10.0, 50.0])
    ax.set_zlabel('Z')

    return ax


def launch(data, is_color=False, N=10000):
    fig = plt.figure()
    fig.set_facecolor("mintcream")

    ax = properties(p3.Axes3D(fig))

    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], color='red')

    anim = animation.FuncAnimation(fig, update, N, fargs=(data, line, is_color), interval=10000 / N, blit=False)

    plt.show()
