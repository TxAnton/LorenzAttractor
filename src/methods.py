import numpy as np


# здесь методы, которыми могут пользоваться другие люди

# метод Эйлера - 1 порядка
def EUL1(step, num_steps, point, func, savePoint=None):
    for i in range(num_steps):
        value_func = func(point)

        for j in range(len(point)):
            point[j] = point[j] + step * value_func[j]

        if savePoint is not None:
            savePoint(point, step * (i + 1), i + 1)

    return np.hstack(point)

# метод средней точки - 2 порядка
def MIDP2(step, num_steps, point, func, savePoint=None):
    temp_point = np.empty(len(point))

    for i in range(num_steps):
        value_func = func(point)

        for j in range(len(point)):
            temp_point[j] = point[j] + (step / 2) * value_func[j]
        value_func = func(temp_point)

        for k in range(len(point)):
            point[k] = point[k] + step * value_func[k]

        if savePoint is not None:
            savePoint(point, step * (i + 1), i + 1)

    return np.hstack(point)


# Runge Kutta 4
def RK4(step, num_steps, point, func, savePoint=None):
    k = np.zeros([4, len(point)])

    for i in range(num_steps):
        k[0] = np.hstack(func(np.asarray(point)))
        k[1] = np.hstack(func(np.asarray(point) + step * (1 / 2) * k[0]))
        k[2] = np.hstack(func(np.asarray(point) + step * (1 / 2) * k[1]))
        k[3] = np.hstack(func(np.asarray(point) + step * k[2]))
        point = np.asarray(point) + step * (1 / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])

        if savePoint is not None:
            savePoint(point, step * (i + 1), i + 1)

    return point

# Adam Bashforts k = 4
def AB4(step, num_steps, point, func, savePoint=None, ABM5=False):
    const_val = [55 / 24, -59 / 24, 37 / 24, -3 / 8]
    points = []
    index = 3

    points.append(point)  # всегда хранятся только 4 точки

    for i in range(index):  # разгон
        new_point = RK4(step, 1, points[i], func, None)
        points.append(new_point)
        if savePoint is not None:  # сохраняем здесь, иначе сбиваются индексы
            savePoint(new_point, step * (i + 1), i + 1)

    for j in range(index, num_steps):
        temp = const_val[0] * np.hstack(func(points[index])) + const_val[1] * np.hstack(func(points[index - 1])) + \
           const_val[2] * np.hstack(func(points[index - 2])) + const_val[3] * np.hstack(func(points[index - 3]))
        points.append(np.array(points[index]) + step * temp)

        if ABM5:
            const_val = [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]

            for i in range(2):
                temp = const_val[0] * np.hstack(func(points[index + 1])) + const_val[1] * np.hstack(
                    func(points[index])) + \
                       const_val[2] * np.hstack(func(points[index - 1])) + const_val[3] * np.hstack(
                    func(points[index - 2])) + \
                       const_val[4] * np.hstack(func(points[index - 3]))
                points[index + 1] = np.array(points[index]) + step * temp

        if savePoint is not None:
            savePoint(points[index + 1], step * (j + 1), j + 1)

        points.pop(0)

    return points[index]


# Adam Moulton 4
def AM4(step, num_steps, point, func, iterations, savePoint=None):
    const_val = [3 / 8, 19 / 24, -5 / 24, 1 / 24]
    points = []
    index = 3

    points.append(point)  # всегда хранятся только 4 точки
    for i in range(index):  # разгон
        new_point = RK4(step, 1, points[i], func, None)
        points.append(new_point)
        if savePoint is not None:  # сохраняем здесь, иначе сбиваются индексы
            savePoint(new_point, step * (i + 1), i + 1)

    for j in range(index, num_steps + 1):
        for i in range(iterations):  # EC
            temp = const_val[0] * np.hstack(func(points[index])) + const_val[1] * np.hstack(func(points[index - 1])) + \
                   const_val[2] * np.hstack(func(points[index - 2])) + const_val[3] * np.hstack(func(points[index - 3]))
            points[index] = np.array(points[index - 1]) + step * temp

        if savePoint is not None:  # сохраняем здесь, иначе сбиваются индексы
            savePoint(points[index], step * j, j)
        points.pop(0)
        points.append(RK4(step, 1, points[index - 1], func, None))
    return points[index]
