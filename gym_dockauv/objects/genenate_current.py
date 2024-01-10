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
if False:
# if True:
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            for k in range(grid_size_z):
                # Distance from the point to the parabola (ignoring the z-coordinate)
                distance = np.abs(Y[i, j, k] - parabola(X[i, j, k]))

                # Direction of the current (derivative of the parabola, ignoring z)
                direction = np.array([1, 0.5 * (X[i, j, k] - A[0] + X[i, j, k] - B[0])])
                direction /= np.linalg.norm(direction)

                # Magnitude of the current decreases with distance from the parabola
                magnitude = (0.5 + k / grid_size_z) * np.exp(-distance) + 0.3
                if magnitude > 1:
                    # raise ValueError("Magnitude is too large")
                    magnitude = 1
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


skip = (slice(None, None, 20), slice(None, None, 20), slice(None, None, 200))

if False:
# if True:
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
else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    curve_len =2
    # 遍历每个点并绘制从该点出发的线段
    for i in range(0, grid_size_x, 20):
        for j in range(0, grid_size_y, 20):
            # for k in range(0, grid_size_z, 200):
            for k in [99]:
                # 线段的起点
                start_point = [XX[i, j, k], YY[i, j, k], ZZ[i, j, k]]

                # 线段的终点
                end_point = [XX[i, j, k] + curve_len * U[i, j, k],
                             YY[i, j, k] + curve_len * V[i, j, k],
                             ZZ[i, j, k] + curve_len * W[i, j, k]]

                # 绘制线段
                ax.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color='blue')

    # 设置标签和标题
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Ocean Current Matrix')

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

#todo dai 改了一天的bug，把y放到了x的位置，x放到了y的位置，为什么呀，正不明白阿！！！！！😎
def generate_current( input_y, input_x,input_z, t):  # longitude (经度), latitude (纬度), altitude (海拔)。
    # try:
    #     assert bool(float(input_x) > float(lon.min()))
    #     assert bool(float(input_x) < float(lon.max()))
    #     assert bool(float(input_y) > float(lat.min()))
    #     assert bool(float(input_y) < float(lat.max()))
    #     assert bool(float(input_z) > float(alt.min()))
    #     assert bool(float(input_z) < float(alt.max()))
    # except:
    #     print("输入的经纬度不在范围内")
    #
    #     print("_lon:", input_x, "lon.min:", lon.min(), "lon.max:", lon.max())
    #     print("_lat:", input_y, "lat.min:", lat.min(), "lat.max:", lat.max())
    #     print("_alt:", input_z, "alt.min:", alt.min(), "alt.max:", alt.max())

    # 哪个更快？
    #time1 4.00543212890625e-05
    # time2 2.1457672119140625e-05
    #amazing

    # t = time.time()
    ind_x = sum(input_x >= lon) - 1
    ind_y = sum(input_y >= lat) - 1
    ind_z = sum(input_z >= alt) - 1
    u = U[ind_x, ind_y, ind_z]
    v = V[ind_x, ind_y, ind_z]
    w = W[ind_x, ind_y, ind_z]
    # print("time1", time.time() - t)

    # # t = time.time()
    # nearest_x_idx = find_nearest_index(lon, input_x)
    # nearest_y_idx = find_nearest_index(lat, input_y)
    # nearest_z_idx = find_nearest_index(alt, input_z)
    #
    # # Retrieve the ocean current vector at this grid point
    # u = U[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    # v = V[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    # w = W[nearest_x_idx, nearest_y_idx, nearest_z_idx]
    # # print("time2", time.time() - t)

    return np.array([u, v, w]) #todo dai 我乱了，这里是不是应该是u,v,w


import matplotlib.pyplot as plt
import numpy as np


# 创建 x 和 y 的网格
x_ = np.arange(-20, 21, 2)
y_ = np.arange(-20, 21, 2)
X_, Y_ = np.meshgrid(x_, y_)

# 初始化 u 和 v 来存储洋流速度的 x 和 y 分量
atest_U = np.zeros_like(X_,dtype=np.float)
atest_V = np.zeros_like(Y_,dtype=np.float)

# 遍历所有点并计算洋流速度
for i in range(X_.shape[0]):
    for j in range(X_.shape[1]):
        position = (X_[i, j], Y_[i, j], 0)  # z 始终为 0
        current = generate_current(*position, 0)
        atest_U[i, j] = current[0]
        atest_V[i, j] = current[1]

plt.figure(figsize=(10, 10))

# 遍历所有点并绘制洋流速度
for i in range(X_.shape[0]):
    for j in range(X_.shape[1]):
        # 线段的起点
        start_point = [X_[i, j], Y_[i, j]]

        # 线段的终点（表示洋流方向和大小）
        end_point = [X_[i, j] + atest_U[i, j], Y_[i, j] + atest_V[i, j]]

        # 绘制线段
        plt.plot([start_point[0], end_point[0]],
                 [start_point[1], end_point[1]],
                 color='blue')

# 添加标签和标题
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Current Velocity Field')
plt.grid(True)
plt.show()

haha = True


# # 维度1000米： 180*1000/math.pi/6371000
# # 中心（0，0）：-21.5  -119.25  0.009030   0.009650
# # （1000，1000）： -21.49095781898798  -119.24035303257898
# # （-1000，-1000）： -21.5090178878837987  -119.25965219008972
