from src.lorenzMethods import AttractorLorenz

if __name__ =="__main__":

    # Euler
    AL1 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL1.EulerMethod()
    AL1.printDots(15)
    AL1.createPNG("PNG/AL1")

    # Midpoint
    AL2 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL2.midpointMethod()
    AL2.printDots(15)
    AL2.createPNG("PNG/AL2")

    # RK4
    AL3 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL3.RKMethod()
    AL3.printDots(15)
    AL3.createPNG("PNG/AL3")

    # Adams Moulton
    # Adams Basford
    # 8
    #



    # # для метода Адамса-Башфорта нужны первые n шагов RK4
    # def dataForAdamsBasforthMethod(x_dots, y_dots, z_dots, f_x, f_y, f_z, n, cur):
    #     for i in range(cur, n + 1):
    #         arr_x_dots, arr_y_dots, arr_z_dots = rungeKuttaMethod(init_vals=(x_dots[i - 1], y_dots[i - 1], z_dots[i - 1]), num_steps=1)
    #         x_dots[i], y_dots[i], z_dots[i] = arr_x_dots[1], arr_y_dots[1], arr_z_dots[1]  # arr_x_dots = {x0, x1}
    #
    #         f_x[i], f_y[i], f_z[i] = diff(x_dots[i], y_dots[i], z_dots[i])
    #     return x_dots, y_dots, z_dots, f_x, f_y, f_z
    #
    #
    # def adamsBashforthMethod(step, n, f_x, f_y, f_z):
    #     coef = [55/24, 59/24, 37/24, 3/8]
    #     x_dots[n + 1] = x_dots[n] + step * (coef[0] * f_x[n] - coef[1] * f_x[n-1] + coef[2] * f_x[n-2] - coef[3] * f_x[n-3])
    #     y_dots[n + 1] = y_dots[n] + step * (coef[0] * f_y[n] - coef[1] * f_y[n-1] + coef[2] * f_y[n-2] - coef[3] * f_y[n-3])
    #     z_dots[n + 1] = z_dots[n] + step * (coef[0] * f_z[n] - coef[1] * f_z[n-1] + coef[2] * f_z[n-2] - coef[3] * f_z[n-3])
    #     return x_dots, y_dots, z_dots
    #
    # def allDots(init_vals=(0., 1., 1.05), num_steps=10000):
    #     n = 3
    #
    #     x_dots, y_dots, z_dots = createDots(num_steps)
    #     x_dots[0], y_dots[0], z_dots[0] = init_vals
    #
    #     f_x, f_y, f_z = createDots(num_steps)
    #     f_x[0], f_y[0], f_z[0] = diff(x_dots[0], y_dots[0], z_dots[0])
    #
    #     for i in range(4):
    #         x_dots, y_dots, z_dots, f_x, f_y, f_z = dataForAdamsBasforthMethod(x_dots, y_dots, z_dots, f_x, f_y, f_z, n + i, i + 1)
    #         for j in range(10):
    #             print(x_dots[j])
    #         print("-----")


    # x_dots, y_dots, z_dots = rungeKuttaMethod()
    # for i in range(5):
    #     print(x_dots[i])
    # check
    #x_dots, y_dots, z_dots = dataForAdamsBasforthMethod()
    # for i in range(10):
    #     print(x_dots[i])

    #createPNG(x_dots, y_dots, z_dots)
