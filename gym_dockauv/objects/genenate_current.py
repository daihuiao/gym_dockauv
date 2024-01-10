import math
import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Set the size of the grid
grid_size_x = 200
grid_size_y = 200
grid_size_z = 200

x = np.linspace(-10, 10, grid_size_x)
y = np.linspace(-10, 10, grid_size_y)
z = np.linspace(-5, 5, grid_size_z)
X, Y, Z = np.meshgrid(x, y, z)

xx = np.linspace(-20.0, 20.0, grid_size_x)  #
yy = np.linspace(-20.0, 20.0, grid_size_y)
zz = np.linspace(-20.0, 20.0, grid_size_z)
# xx = np.linspace(-119.25 - 0.009650 * 2, -119.25 + 0.009650 * 2, grid_size_x)#
# yy = np.linspace(-21.5 - 0.009030 * 2, -21.5 + 0.009030 * 2, grid_size_y)
# zz = np.linspace(-500, 0, grid_size_z)
XX, YY, ZZ = np.meshgrid(xx, yy, zz)

# Define the parabola
A = np.array([-4, 0])  # Point A
B = np.array([4, 0])  # Point B
parabola = lambda x: 0.25 * (x - A[0]) * (x - B[0])  # Parabolic function

# Calculate the ocean current vector field in 3D
U = np.zeros_like(X)
V = np.zeros_like(Y)
W = np.zeros_like(Z)  # No vertical flow component
# if False:
if True:
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            for k in range(grid_size_z):
                # Distance from the point to the parabola (ignoring the z-coordinate)
                distance = np.abs(Y[i, j, k] - parabola(X[i, j, k]))

                # Direction of the current (derivative of the parabola, ignoring z)
                direction = np.array([1, 0.5 * (X[i, j, k] - A[0] + X[i, j, k] - B[0])])
                direction /= np.linalg.norm(direction)

                # Magnitude of the current decreases with distance from the parabola
                magnitude = (0.5 + k / grid_size_z) * np.exp(-distance) + 0.1
                if magnitude > 2:
                    raise ValueError("Magnitude is too large")
                    magnitude = 0
                # Calculate the vector components
                U[i, j, k] = 1*magnitude * direction[0]
                V[i, j, k] = 1*magnitude * direction[1]

    with open("current.pkl", "wb") as f:
        import pickle

        pickle.dump([U, V, W], f)
else:
    with open("current.pkl", "rb") as f:
        import pickle

        U, V, W = pickle.load(f)

skip = (slice(None, None, 20), slice(None, None, 20), slice(None, None, 10))

# if False:
if True:
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the current arrows
    ax.quiver(XX[skip], YY[skip], ZZ[skip],
              U[skip],
              V[skip],
              W[skip],
              # length=np.sqrt(U[skip]**2+V[skip]**2+W[skip]**2), color='blue')
              length=np.linalg.norm([U[skip], V[skip], W[skip]]), color='blue')

    # # Plot points A and B, and the parabolic path
    # ax.scatter(*A, color='red', s=100, label='Point A')
    # ax.scatter(*B, color='green', s=100, label='Point B')
    # # ax.plot(parabola[:,0], parabola[:,1], parabola[:,2], color='orange', label='Parabolic Path')
    # parabola_3d = np.array([[x, parabola(x), 0] for x in np.linspace(A[0], B[0], 100)])
    # ax.plot(parabola_3d[:, 0], parabola_3d[:, 1], parabola_3d[:, 2], color='orange', label='Parabolic Path')

    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Ocean Current Matrix')
    ax.legend()
    # Show the plot
    plt.show()

lon = xx
lat = yy
alt = zz


def find_nearest_index(array, value):
    """
    Find the index of the nearest value in an array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def generate_current(input_x, input_y, input_z, t):  # longitude (经度), latitude (纬度), altitude (海拔)。
    try:
        assert bool(float(input_x) > float(lon.min()))
        assert bool(float(input_x) < float(lon.max()))
        assert bool(float(input_y) > float(lat.min()))
        assert bool(float(input_y) < float(lat.max()))
        assert bool(float(input_z) > float(alt.min()))
        assert bool(float(input_z) < float(alt.max()))
    except:
        print("输入的经纬度不在范围内")

        print("_lon:", input_x, "lon.min:", lon.min(), "lon.max:", lon.max())
        print("_lat:", input_y, "lat.min:", lat.min(), "lat.max:", lat.max())
        print("_alt:", input_z, "alt.min:", alt.min(), "alt.max:", alt.max())
    # 哪个更快？
    #time1 4.00543212890625e-05
    # time2 2.1457672119140625e-05
    #amazing

    # t = time.time()
    # ind_x = sum(input_x >= lon) - 1
    # ind_y = sum(input_y >= lat) - 1
    # ind_z = sum(input_z >= alt) - 1
    # u = U[ind_x, ind_y, ind_z]
    # v = V[ind_x, ind_y, ind_z]
    # w = W[ind_x, ind_y, ind_z]
    # print("time1", time.time() - t)

    # t = time.time()
    nearest_x_idx = find_nearest_index(X[0, :, 0], input_x)
    nearest_y_idx = find_nearest_index(Y[:, 0, 0], input_y)
    nearest_z_idx = find_nearest_index(Z[0, 0, :], input_z)

    # Retrieve the ocean current vector at this grid point
    u = U[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    v = V[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    w = W[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    # print("time2", time.time() - t)

    return np.array([u, v, w])

# #checking the code
# temp_current = [np.zeros_like(X) , np.zeros_like(Y) , np.zeros_like(Z)]
# lon_temp,lat_temp,alt_temp = lon+0.0001,lat+0.0001,alt+0.0001
#
# for x,lon_ in enumerate(lon_temp[:-1]):
#     for y,lat_ in enumerate(lat_temp[:-1]):
#         for z,alt_ in enumerate(alt_temp[:-1]):
#             c, ha, va, (rho,u,v,w)=gen_current(lon_,lat_,alt_,0)
#             temp_current[0][x,y,z] = u
#             temp_current[1][x,y,z] = v
#             temp_current[2][x,y,z] = w
#
#
# # Create the 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the current arrows
# ax.quiver(XX[skip], YY[skip], ZZ[skip],
#           temp_current[0][skip],
#           temp_current[1][skip],
#           temp_current[2][skip],
#           # length=np.sqrt(U[skip]**2+V[skip]**2+W[skip]**2), color='blue')
#           length=0.00005 * np.linalg.norm([temp_current[0][skip], temp_current[0][skip], temp_current[0][skip]]), color='blue')
#
# # # Plot points A and B, and the parabolic path
# # ax.scatter(*A, color='red', s=100, label='Point A')
# # ax.scatter(*B, color='green', s=100, label='Point B')
# # # ax.plot(parabola[:,0], parabola[:,1], parabola[:,2], color='orange', label='Parabolic Path')
# # parabola_3d = np.array([[x, parabola(x), 0] for x in np.linspace(A[0], B[0], 100)])
# # ax.plot(parabola_3d[:, 0], parabola_3d[:, 1], parabola_3d[:, 2], color='orange', label='Parabolic Path')
#
# # Set labels and title
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('3D Ocean Current Matrix')
# ax.legend()
#
# # Show the plot
# plt.show()
#
# pause = 0
# # 维度1000米： 180*1000/math.pi/6371000
# # 中心（0，0）：-21.5  -119.25  0.009030   0.009650
# # （1000，1000）： -21.49095781898798  -119.24035303257898
# # （-1000，-1000）： -21.5090178878837987  -119.25965219008972
