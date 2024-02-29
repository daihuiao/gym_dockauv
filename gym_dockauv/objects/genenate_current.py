import math
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import copy
import timeit
import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.polynomial import polyfit
import plotly.figure_factory as ff


def find_nearest_index(array, value):
    """
    Find the index of the nearest value in an array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if_draw = False


class Karmen_current():
    draw = False

    def __init__(self, range_x=26, range_y=18, range_z=20, start_point=[-0, -10], goal_point=[-0, 10],
                 current_index=250):
        self.start_point = start_point
        self.goal_point = goal_point
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        # Set the size of the grid
        self.grid_size_x = 2 * range_x
        self.grid_size_y = 2 * range_y
        self.grid_size_z = 2 * range_z


        project_path = pathlib.Path(__file__).parent.parent.parent.parent.__str__()

        # with open("/home/ps/dai/overall/togithub/gym_dockauv/karmen_520_180.pkl", "rb") as f:
        with open(project_path+"/gym_dockauv/karmen_520_360.pkl", "rb") as f:
            import pickle
            self.Us = pickle.load(f)
            # Create streamline figure
        self.u = 10 * copy.deepcopy(self.Us[current_index])
        self.U = self.u[0, :, :].transpose()
        self.V = self.u[1, :, :].transpose()

        self.xx = np.linspace(-range_x, range_x, self.U.shape[1])  #
        self.yy = np.linspace(-range_y, range_y, self.U.shape[0])
        self.zz = np.linspace(-range_z, range_z, self.grid_size_z)
        self.XX, self.YY, self.ZZ = np.meshgrid(self.xx, self.yy, self.zz)

        # Coordinates of the circular obstacle
        self.cx = 2 * range_x / 4
        self.cy = 2 * range_y / 2
        self.r = 2 * range_y / 9

    def draw_current(self):
        if False:
        # if True:
            w = self.u[0, :, :].transpose()
            v = self.u[1, :, :].transpose()
            x = np.arange(0, self.grid_size_x, self.grid_size_x / w.shape[1])
            y = np.arange(0, self.grid_size_y, self.grid_size_y / v.shape[0])
            circle = plt.Circle((self.cx, self.cy), self.r, color='blue')
            fig, ax = plt.subplots()
            ax.add_artist(circle)
            d = 1
            plt.streamplot(x, y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
            ax.set_aspect('equal')
            plt.savefig('streamlines.png', bbox_inches='tight', dpi=20)
            plt.show()

            plt.imshow(np.sqrt(self.u[0] ** 2 + self.u[1] ** 2).transpose(),
                       cmap=cm.Greens)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.
            # print(Iter)
            # plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight', dpi=200)
            plt.show()

    def generate_rainbow_color_scheme(self,color_count):

        import colorsys

        def hsv_to_rgb(h, s, v):
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # r = int(r )
            # g = int(g * 255)
            # b = int(b * 255)
            return r, g, b
        # 确定每个色带的步长
        step = 360 / color_count

        colors = []
        for i in range(color_count):
            # 计算当前颜色的色相
            hue = int(200+(i * step)/360*100)
            # 将色相转换为RGB值
            rgb = hsv_to_rgb(hue / 360, 1, 1)
            # 将RGB值添加到颜色列表中
            colors.append(rgb)

        return colors
    def trajectory_in_current(self, position, prefix):
        import matplotlib.colors as mcolors
        position_scale  = 1
        w = self.u[0, :, :].transpose()
        v = self.u[1, :, :].transpose()
        x = np.arange(0 - self.grid_size_x / 2, 0 + self.grid_size_x / 2, self.grid_size_x / w.shape[1])
        y = np.arange(0 - self.grid_size_y / 2, 0 + self.grid_size_y / 2, self.grid_size_y / v.shape[0])
        circle = plt.Circle((position_scale * self.cx - position_scale * self.grid_size_x / 2,
                             position_scale * self.cy - position_scale * self.grid_size_y / 2),
                            position_scale * self.r, color='blue')
        fig, ax = plt.subplots()
        ax.add_artist(circle)
        d = 2
        plt.streamplot(position_scale*x, position_scale*y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
        ax.set_aspect('equal')
        colors = self.generate_rainbow_color_scheme(len(position))
        if len(position[0])>5:
            for i in range(len(position)):
                position[i] = [position_scale * position[i][j] for j in range(len(position[i]))]
                color = mcolors.rgb2hex(colors[i])  # 将RGB值转换为十六进制颜色字符串
                ax.plot(np.array(position[i])[:-10, 0], np.array(position[i])[:-10, 1], color=color)
                plt.scatter(position[i][0][0], position[i][0][1], c=color)
        else:
            ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], c="orange")
            plt.scatter(self.start_point[0], self.start_point[1], c="r")
            plt.scatter(self.goal_point[0], self.goal_point[1], c="g")
        plt.savefig("thruster_off.png", bbox_inches='tight', dpi=200)
        if prefix is None:
            plt.show()
        else:
            plt.savefig(prefix, bbox_inches='tight', dpi=200)
        # plt.show()
        plt.cla()
        plt.clf()

        if False:
            fig, ax = plt.subplots()
            # ax.plot(np.array(positon)[:,0], np.array(positon)[:,1])
            im = ax.imshow(np.sqrt(self.u[0] ** 2 + self.u[1] ** 2).transpose(),
                       extent=[-260, 260, -180, 180],cmap=cm.Blues)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Y(m)')
            ax.set_xlim(-260, 260)
            # ax.set_title('Position and Velocity Field')
            cbar = fig.colorbar(im,shrink=0.7)
            cbar.set_label('Velocity Magnitude')
            # ax.legend()
            # print(Iter)
            # plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight', dpi=200)
            plt.savefig("Velocity Magnitude.png", bbox_inches='tight', dpi=200)
            plt.show()
            haha = True

    def generate_current(self, input_x, input_y, input_z, t):  # longitude (经度), latitude (纬度), altitude (海拔)。
        try:
            assert bool(float(input_x) > float(self.xx.min()))
            assert bool(float(input_x) < float(self.xx.max()))
            assert bool(float(input_y) > float(self.yy.min()))
            assert bool(float(input_y) < float(self.yy.max()))
            assert bool(float(input_z) > float(self.zz.min()))
            assert bool(float(input_z) < float(self.zz.max()))
        except:
            print("输入的经纬度不在范围内")

            print("_lon:", input_x, "lon.min:", self.xx.min(), "lon.max:", self.xx.max())
            print("_lat:", input_y, "lat.min:", self.yy.min(), "lat.max:", self.yy.max())
            print("_alt:", input_z, "alt.min:", self.zz.min(), "alt.max:", self.zz.max())

        ind_x = sum(input_x >= self.xx) - 1
        ind_y = sum(input_y >= self.yy) - 1
        u = self.U[ind_y, ind_x]
        v = self.V[ind_y, ind_x]
        w = 0

        return np.array([u, v, w])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(200,300,10):
        i = i
        karmen_current = Karmen_current(current_index=i)
        print(i)
        karmen_current.draw_current()
        if False:
            # 创建 x 和 y 的网格
            x_ = np.arange(-karmen_current.range_x, karmen_current.range_x, 5)
            y_ = np.arange(-karmen_current.range_y, karmen_current.range_y, 5)
            X_, Y_ = np.meshgrid(x_, y_)

            # 初始化 u 和 v 来存储洋流速度的 x 和 y 分量
            atest_U = np.zeros_like(X_, dtype=np.float)
            atest_V = np.zeros_like(Y_, dtype=np.float)

            # 遍历所有点并计算洋流速度
            max = 0
            max_point = [0, 0]
            strange_current = []
            for i in range(X_.shape[0]):
                for j in range(X_.shape[1]):
                    position = (X_[i, j], Y_[i, j], 0)  # z 始终为 0
                    current = karmen_current.generate_current(*position, 0)
                    if np.linalg.norm(position - np.array([-karmen_current.cx, 0, 0])) < karmen_current.r:
                        strange_current.append(current)
                        current = np.array([0, 0, 0])

                    if np.linalg.norm(current) > max:
                        max = np.linalg.norm(current)
                        max_point = position
                    atest_U[i, j] = current[0]
                    atest_V[i, j] = current[1]

            plt.figure(figsize=(10, 10))
            point_len = 10
            # 遍历所有点并绘制洋流速度
            for i in range(X_.shape[0]):
                for j in range(X_.shape[1]):
                    # 线段的起点
                    start_point = [X_[i, j], Y_[i, j]]

                    # 线段的终点（表示洋流方向和大小）
                    end_point = [X_[i, j] + point_len * atest_U[i, j], Y_[i, j] + point_len * atest_V[i, j]]

                    # 绘制线段
                    plt.plot([start_point[0], end_point[0]],
                             [start_point[1], end_point[1]],
                             color='blue')
            plt.axis('equal')
            # 添加标签和标题
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Current Velocity Field')
            plt.grid(True)

            plt.show()

    haha = True
