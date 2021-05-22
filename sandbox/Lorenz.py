"""
================
Lorenz Attractor
================
"""
import numpy as np
import matplotlib.pyplot as plt


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


# # dt = 0.0248872675
# dt = 0.01
# num_steps = 10000

def gen_points(init_vals=(10., 1., 1.05), num_steps=10000, dt=0.01, s=10, r=28, b=2.667):
    # Need one more for the initial values

    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = init_vals

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    dots = [xs, ys, zs]
    return dots


# по методу эйлера
def gen_points_optimal(init_vals=(0., 1., 1.05), num_steps=10000, dt=0.01, s=10, r=28, b=2.667):
    # Need one more for the initial values

    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    ts = np.empty(num_steps + 1)
    # Set initial values
    xs[0], ys[0], zs[0] = init_vals
    ts[0] = 0

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (dt / 2) * (x_dot)  # x_1=x_0+h*f(x_0, y_0, z_0)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        x_dot2, y_dot2, z_dot2 = lorenz(xs[i + 1], ys[i + 1], zs[i + 1], s, r, b)
        xs[i + 1] = xs[i] + (dt / 2) * (x_dot + x_dot2)
        ys[i + 1] = ys[i] + (dt / 2) * (y_dot + y_dot2)
        zs[i + 1] = zs[i] + (dt / 2) * (z_dot + z_dot2)
        ts[i + 1] = ts[i] + dt

    dots = [xs, ys, zs, ts]
    return dots


# Plot

# # Beauty
# s=10
# r=28
# b=2.667

# #I1 +
# s=10
# r=28
# b=2*s

# I2
s = 10
r = 0
b = 1

# # I3
# s = 1
# r = 10
# b = 1

# #I4
# s=.500000
# r=2.0 * s - 1.0
# b=6.0*s - 2.0

# #I5 +
# s=1
# r=28
# b=4

xs, ys, zs, ts = gen_points_optimal(s=s, r=r, b=b)

m = np.vstack([xs, ys, zs, ts])

# I1 = (m[0]**2-2*s*m[2])*np.exp(2*s*m[3]) # req b==2s
#
I2 = (m[1]**2 + m[2]**2 )*np.exp(2*m[3]) # req b=1, r=0
#
# I3 = (-r**2 * m[0]**2 + m[1]**2 + m[2]**2)**np.exp(2*m[3]) # req b = 1, s = 1

# I4 = ((((2.0 * s - 1) ** 2) * s) *
#       m[0] ** 2 + s * m[1] ** 2 - \
#       (4 * s - 2) * m[0] * m[1] - \
#       (1 / (4 * s)) * m[0] ** 4 + \
#       m[0] ** 2 * m[2]) * \
#      np.exp(4 * s * m[3])  # req b = 6*s - 2, r = 2*s-1

# I5 = ( -r*m[0]**2 - m[1]**2 + 2*m[0]*m[1]  + 0.25*m[0]**4 - m[0]**2*m[2] + 4*(r-1)*m[2])*np.exp(4*m[3]) # req b = 4, s = 1

# minf=I1[-1] # -inf
mm = -1  # [i for i in range(len(I1)) if I1[i]==minf][0]

# plt.plot(m[3][:mm],I3[:mm])
plt.plot(I2[:mm])
plt.show()
print("WAIT")
exit(0)
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
# plt.savefig('Lorenz Attractor')
