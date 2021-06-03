import numpy as np
# здесь методы, которыми могут пользоваться другие люди

# метод Эйлера - 1 порядка
def EUL1(step, dots, func):
    value_func = func(dots)
    next_dots = []
    for i in range(len(dots)):
        next_dots.append(dots[i] + step * value_func[i])
    return next_dots


# метод средней точки - 2 порядка
def MIDP2(step, dots, func):
    value_func = func(dots)
    next_dots = []
    for i in range(len(dots)):
        next_dots.append(dots[i] + (step / 2) * value_func[i])

    value_func = func(next_dots)
    for i in range(len(dots)):
        next_dots[i] = dots[i] + step * value_func[i]
    return next_dots


# Runge Kutta 4
def RK4(step, dots, func):
    k = np.zeros([4, len(dots)])
    k[0] = np.hstack(func(np.asarray(dots)))
    k[1] = np.hstack(func(np.asarray(dots) + step * (1 / 2) * k[0]))
    k[2] = np.hstack(func(np.asarray(dots) + step * (1 / 2) * k[1]))
    k[3] = np.hstack(func(np.asarray(dots) + step * k[2]))
    return np.asarray(dots) + step * (1 / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])


# Adam Bashforts k = 4
def AB4(step, dots, func):
    index = len(dots) - 1
    const_val = [55 / 24, -59 / 24, 37 / 24, -3 / 8]

    temp = const_val[0] * np.hstack(func(dots[index])) + const_val[1] * np.hstack(func(dots[index - 1])) + \
           const_val[2] * np.hstack(func(dots[index - 2])) + const_val[3] * np.hstack(func(dots[index - 3]))
    return np.array(dots[index]) + step * temp


# Adam Moulton 4
def AM4(step, dots, func, iterations):
    index = len(dots) - 1
    const_val = [3 / 8, 19 / 24, -5 / 24, 1 / 24]

    for i in range(iterations):
        temp = const_val[0] * np.hstack(func(dots[index])) + const_val[1] * np.hstack(func(dots[index - 1])) + \
               const_val[2] * np.hstack(func(dots[index - 2])) + const_val[3] * np.hstack(func(dots[index - 3]))
        dots[index] = np.array(dots[index - 1]) + step * temp
    return dots[index]


def ABM5(step, dots, func, iterations):  # 0 1 2 3 4
    index = len(dots) - 1
    const_val = [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]
    for i in range(iterations):
        temp = const_val[0] * np.hstack(func(dots[index])) + const_val[1] * np.hstack(func(dots[index - 1])) + \
               const_val[2] * np.hstack(func(dots[index - 2])) + const_val[3] * np.hstack(func(dots[index - 3])) + \
               const_val[4] * np.hstack(func(dots[index - 4]))
        dots[index] = np.array(dots[index - 1]) + step * temp
    return dots[index]
