# """
# 我要生成一个三维洋流矩阵，在这个矩阵里有两个点A和B，其中直接从A点到B点的直线上洋流小，而旁边一条曲线上从A到B的洋流大，在空间中其他位置根据这两条线生成，变化过度自然一点。请给出python代码，感谢
# 使用箭头表示洋流方向,画图的时候只画十分之一的点,否则可能看不清。
#
# """
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def create_ocean_current_matrix(dimensions, point_a, point_b, strength=1.0, width=0.1):
#     """
#     创建一个三维洋流矩阵，其中 A 和 B 之间的直线上洋流较小，旁边曲线上洋流较大。
#
#     :param dimensions: 三维空间的大小 (x, y, z)
#     :param point_a: 点 A 的坐标 (x, y, z)
#     :param point_b: 点 B 的坐标 (x, y, z)
#     :param strength: 洋流的最大强度
#     :param width: 影响洋流大小的宽度因子
#     :return: 三维洋流矩阵
#     """
#     x, y, z = np.indices(dimensions)
#     ocean_current = np.zeros(dimensions)
#
#     # 创建直线 AB
#     line_vector = np.array(point_b,dtype=np.float64) - np.array(point_a,dtype=np.float64)
#     line_length = np.linalg.norm(line_vector)
#     line_vector /= line_length
#
#     # 计算每个点到直线 AB 的最短距离
#     for i in range(dimensions[0]):
#         for j in range(dimensions[1]):
#             for k in range(dimensions[2]):
#                 point = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
#                 vector_ap = point - np.array(point_a)
#                 distance_to_line = np.linalg.norm(vector_ap - np.dot(vector_ap, line_vector) * line_vector)
#
#                 # 使用高斯函数来设置洋流强度
#                 ocean_current[i, j, k] = strength * np.exp(-distance_to_line ** 2 / (2 * width ** 2))
#
#     return ocean_current
#
#
# # 设置参数
# dimensions = (50, 50, 50)  # 三维空间的大小
# point_a = (10, 10, 25)  # 点 A
# point_b = (40, 40, 25)  # 点 B
#
# # # 生成洋流矩阵
# # ocean_current = create_ocean_current_matrix(dimensions, point_a, point_b)
# #
# # # 可视化（可选）
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x, y, z = np.indices(dimensions)
# # ax.scatter(x[::5,::5,::5], y[::5,::5,::5], z[::5,::5,::5], c=ocean_current[::5,::5,::5].flatten(), alpha=0.1)
# # ax.scatter(*point_a, color='red')  # 点 A
# # ax.scatter(*point_b, color='green')  # 点 B
# # plt.show()
#
# # 生成洋流矩阵
# ocean_current = create_ocean_current_matrix(dimensions, point_a, point_b)
#
# # 计算洋流的梯度
# grad_x, grad_y, grad_z = np.gradient(ocean_current)
#
# # 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 选择一小部分点来显示箭头
# skip = (slice(None, None, 5), slice(None, None, 5), slice(None, None, 5))
# x, y, z = np.indices(dimensions)[skip]
# u, v, w = grad_x[skip], grad_y[skip], grad_z[skip]
#
# # 绘制箭头
# ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
#
# # 标记点 A 和 B
# ax.scatter(*point_a, color='red', s=50)  # 点 A
# ax.scatter(*point_b, color='green', s=50)  # 点 B
#
# plt.show()
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义三维空间大小
dim = 50

# 创建三维洋流矩阵
currents = np.zeros((dim, dim, dim, 3))  # 最后一个维度是洋流的向量

# 定义点A和B
A = np.array([10, 10, 10])
B = np.array([40, 40, 10])

# 生成洋流方向和强度
direction = (B - A)
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            point = np.array([i, j, k])
            distance_to_line = np.linalg.norm(np.cross(direction, A - point)) / np.linalg.norm(direction)
            assert distance_to_line >= 0
            influence = np.exp(-distance_to_line / 5)  # 距离越远影响越小
            currents[i, j, k, :] = direction /np.linalg.norm(B - A) * influence

# 可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 由于点很多，只绘制一部分
skip = 10
for i in range(0, dim, skip):
    for j in range(0, dim, skip):
        for k in range(0, dim, skip):
            x, y, z = i, j, k
            u, v, w = currents[i, j, k]
            ax.quiver(x, y, z, u, v, w, length=5*np.linalg.norm(currents[i,j,k]), normalize=True)

ax.set_xlim([0, dim])
ax.set_ylim([0, dim])
ax.set_zlim([0, dim])
plt.title('3D Ocean Currents Visualization')
plt.show()


