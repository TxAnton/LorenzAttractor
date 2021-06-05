import numpy as np


# здесь методы, которыми могут пользоваться другие люди

# метод Эйлера - 1 порядка
def EUL1(step, points, func):
    value_func = func(points)
    next_points = []
    for i in range(len(points)):
        next_points.append(points[i] + step * value_func[i])
    return next_points


# метод средней точки - 2 порядка
def MIDP2(step, points, func):
    value_func = func(points)
    next_points = []
    for i in range(len(points)):
        next_points.append(points[i] + (step / 2) * value_func[i])

    value_func = func(next_points)
    for i in range(len(points)):
        next_points[i] = points[i] + step * value_func[i]
    return next_points


# Runge Kutta 4
def RK4(step, points, func):
    k = np.zeros([4, len(points)])
    k[0] = np.hstack(func(np.asarray(points)))
    k[1] = np.hstack(func(np.asarray(points) + step * (1 / 2) * k[0]))
    k[2] = np.hstack(func(np.asarray(points) + step * (1 / 2) * k[1]))
    k[3] = np.hstack(func(np.asarray(points) + step * k[2]))
    return np.asarray(points) + step * (1 / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])


# Adam Bashforts k = 4
def AB4(step, points, func):
    index = len(points) - 1
    const_val = [55 / 24, -59 / 24, 37 / 24, -3 / 8]

    temp = const_val[0] * np.hstack(func(points[index])) + const_val[1] * np.hstack(func(points[index - 1])) + \
           const_val[2] * np.hstack(func(points[index - 2])) + const_val[3] * np.hstack(func(points[index - 3]))
    return np.array(points[index]) + step * temp


# Adam Moulton 4
def AM4(step, points, func, iterations):
    index = len(points) - 1
    const_val = [3 / 8, 19 / 24, -5 / 24, 1 / 24]

    for i in range(iterations):  # EC
        temp = const_val[0] * np.hstack(func(points[index])) + const_val[1] * np.hstack(func(points[index - 1])) + \
               const_val[2] * np.hstack(func(points[index - 2])) + const_val[3] * np.hstack(func(points[index - 3]))
        points[index] = np.array(points[index - 1]) + step * temp
    return points[index]


def ABM5(step, points, func, iterations):  # 0 1 2 3 4
    index = len(points) - 1
    const_val = [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]
    for i in range(iterations):
        temp = const_val[0] * np.hstack(func(points[index])) + const_val[1] * np.hstack(func(points[index - 1])) + \
               const_val[2] * np.hstack(func(points[index - 2])) + const_val[3] * np.hstack(func(points[index - 3])) + \
               const_val[4] * np.hstack(func(points[index - 4]))
        points[index] = np.array(points[index - 1]) + step * temp
    return points[index]

