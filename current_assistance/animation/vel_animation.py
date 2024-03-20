import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

position_scale = 1


class Karmen_current():
    draw = False

    def __init__(self, range_x=26, range_y=18, range_z=20, start_point=[-0, -5], goal_point=[-0, 5],
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
        with open(project_path + "/gym_dockauv/current/karmen_520_360.pkl", "rb") as f:
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
        position_scale = 10
        # if False:
        if True:
            w = self.u[0, :, :].transpose()
            v = self.u[1, :, :].transpose()
            x = np.arange(0, self.grid_size_x, self.grid_size_x / w.shape[1])
            y = np.arange(0, self.grid_size_y, self.grid_size_y / v.shape[0])

            circle = plt.Circle((position_scale * self.cx, position_scale * self.cy), position_scale * self.r, color='blue')
            fig, ax = plt.subplots()
            ax.add_artist(circle)
            d = 1
            plt.streamplot(position_scale * x, position_scale * y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
            ax.set_aspect('equal')
            # plt.savefig('streamlines.png', bbox_inches='tight', dpi=20)
            plt.show()

            plt.imshow(np.sqrt(self.u[0] ** 2 + self.u[1] ** 2).transpose(),
                       cmap=cm.Greens)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.
            # print(Iter)
            # plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight', dpi=200)
            plt.show()
    def draw_current_1(self):
        position_scale = 10
        # if False:
        if True:
            w = self.u[0, :, :].transpose()
            v = self.u[1, :, :].transpose()
            x = np.arange(0, self.grid_size_x, self.grid_size_x / w.shape[1])
            y = np.arange(0, self.grid_size_y, self.grid_size_y / v.shape[0])

            circle = plt.Circle((position_scale * self.cx, position_scale * self.cy), position_scale * self.r, color='blue')
            fig, ax = plt.subplots()
            ax.add_artist(circle)
            d = 1
            plt.streamplot(position_scale * x, position_scale * y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
            ax.set_aspect('equal')
            # plt.savefig('streamlines.png', bbox_inches='tight', dpi=20)
            plt.show()

            # plt.imshow(np.sqrt(self.u[0] ** 2 + self.u[1] ** 2).transpose(),
            #            cmap=cm.Greens)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.
            # # print(Iter)
            # # plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight', dpi=200)
            # plt.show()

# 创建Karmen_current对象
karmen_current = Karmen_current()

# 设置动画图像尺寸
fig, ax = plt.subplots(figsize=(8, 6))

# 添加圆形障碍物
circle = plt.Circle(((karmen_current.cx - karmen_current.range_x) * position_scale,
                     (karmen_current.cy -  karmen_current.range_y) * position_scale),
                    karmen_current.r * position_scale, color='blue', fill=True)
ax.add_artist(circle)
ax.set_xlim([-260, 260])
ax.set_ylim([-180, 180])
ax.set_aspect('equal')


plt.imshow(np.sqrt(karmen_current.u[0] ** 2 + karmen_current.u[1] ** 2).transpose(),
           cmap=cm.Greens)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.

# 更新流线图的函数
def update(index):
    print("index:",index)
    karmen_current.u = 10 * karmen_current.Us[index]
    karmen_current.U = karmen_current.u[0, :, :].transpose()
    karmen_current.V = karmen_current.u[1, :, :].transpose()

    # 清除轴上的所有艺术家对象
    ax.cla()
    # 移除旧的流线图
    if ax.collections:
        for coll in ax.collections:
            coll.remove()

    # 添加圆形障碍物
    # ax.add_artist(circle)
    # 绘制新的流线图
    # stream = ax.streamplot(karmen_current.xx * position_scale, karmen_current.yy * position_scale, karmen_current.U, karmen_current.V,
    #                        density=1, linewidth=1, arrowsize=1, color='r')

    stream = ax.imshow(np.sqrt(karmen_current.u[0] ** 2 + karmen_current.u[1] ** 2).transpose(),
               cmap=cm.Greens)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.

    return stream

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(karmen_current.Us), interval=10, blit=False)
ani.save('vel.gif', writer='imagemagick', fps=15)

# 显示动画
# plt.show()