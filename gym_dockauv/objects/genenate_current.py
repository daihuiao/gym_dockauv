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
    def trajectory_in_current_(self, position, prefix, args=None,position1=None,args1=None):
        import matplotlib.colors as mcolors
        plt.rcParams['text.usetex'] = True
        if args is not None:
            position_scale = 10
        else:
            position_scale = 1
        w = self.u[0, :, :].transpose()
        v = self.u[1, :, :].transpose()
        x = np.arange(0 - self.grid_size_x / 2, 0 + self.grid_size_x / 2, self.grid_size_x / w.shape[1])
        y = np.arange(0 - self.grid_size_y / 2, 0 + self.grid_size_y / 2, self.grid_size_y / v.shape[0])
        circle = plt.Circle((position_scale * self.cx - position_scale * self.grid_size_x / 2,
                             position_scale * self.cy - position_scale * self.grid_size_y / 2),
                            position_scale * self.r, color='blue')
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.add_artist(circle)
        d = 2
        plt.streamplot(position_scale * x, position_scale * y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
        ax.set_aspect('equal')
        colors = self.generate_rainbow_color_scheme(len(position))
        """
        step1
        """

        labels = args["label"]
        annotated = args["annotated"]
        for i in range(len(position)):
            position[i] = [position_scale * position[i][j] for j in range(len(position[i]))]
            color = mcolors.rgb2hex(colors[i])  # 将RGB值转换为十六进制颜色字符串
            # if not i==1:
            #     ax.plot(np.array(position[i])[:-10, 0], np.array(position[i])[:-10, 1]
            #             , color=color, label=labels[i])
            plt.scatter(position[i][0][0], position[i][0][1], c=color)
            half_len = int(len(position[i]) / 2)
            plt.annotate(annotated[i], (position[i][half_len][0], position[i][half_len][1]),
                         textcoords="offset points", xytext=((-1) ** i * 60, -75), ha='center' ,fontsize='small',
                         bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.1'))
        # plt.text(args["start_point"][0] - 50, args["start_point"][1] - 25, "Source Point",
        #          fontsize=18,)  # 添加起始点标记及文字说明
        # plt.text(args["goal_point"][0] - 75, args["goal_point"][1] + 25, "Destination Point",
        #          fontsize=16)  # 添加起始点标记及文字说明
        plt.scatter(args["start_point"][0], args["start_point"][1], c="r", label="Starting point")
        plt.scatter(args["goal_point"][0], args["goal_point"][1], c="g", label="Target point")

        """step2"""
        position1 = [position_scale * position1[j] for j in range(len(position1))]

        # cmap = cm.get_cmap('jet')  viridis coolwarm cool RdYlBu  Set1 Set2 Pastel
        cmap = cm.get_cmap('cool')  # 使用预定义的颜色映射，例如'jet'

        scatter = ax.scatter(np.concatenate((np.array(position1)[:, 0],np.array(position[0])[:, 0])),
                             np.concatenate((np.array(position1)[:, 1],np.array(position[0])[:, 1])),
                             c=np.concatenate((args1.get("cmap_value", "orange"),args1.get("cmap_value_1", "orange"))) / 1000.,
                             cmap=cmap,
                             sizes=4 * np.ones(len(position1))
                             )  # 使用速度作为颜色，viridis是一个预定义的颜色映射
        ax.plot(np.array(position[0])[:, 0], np.array(position[0])[:, 1],"k--",label=labels[0])  # 绘制轨迹线，使用黑色虚线
        ax.plot(np.array(position1)[:, 0], np.array(position1)[:, 1],"k:",label=labels[1])  # 绘制轨迹线，使用黑色虚线

        # ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], 'k--', c=cmap(args.get("cmap_value", "orange")/1000.))
        # 设置颜色条
        cbar = plt.colorbar(scatter,shrink=0.6)
        cbar.set_label('Thruster Speed $10^3 $')
        # plt.scatter(self.start_point[0], self.start_point[1], c="r")
        # plt.scatter(self.goal_point[0], self.goal_point[1], c="g")
        try:
            plt.savefig(f"./draw/fig/color/推进器转速3.png", bbox_inches='tight', dpi=200)
        except:
            plt.savefig(f"../fig/color/推进器转速3.png", bbox_inches='tight', dpi=200)
        plt.xlabel('X(m)',fontsize='small')
        plt.ylabel('Y(m)',fontsize='small')
        plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.8),fontsize='x-small')
        plt.show()
        haha = True


    def trajectory_in_current(self, position, prefix, args=None):

        import matplotlib.colors as mcolors
        plt.rcParams['text.usetex'] = True
        if args is not None:
            position_scale = 10
        else:
            position_scale = 1
        w = self.u[0, :, :].transpose()
        v = self.u[1, :, :].transpose()
        x = np.arange(0 - self.grid_size_x / 2, 0 + self.grid_size_x / 2, self.grid_size_x / w.shape[1])
        y = np.arange(0 - self.grid_size_y / 2, 0 + self.grid_size_y / 2, self.grid_size_y / v.shape[0])
        circle = plt.Circle((position_scale * self.cx - position_scale * self.grid_size_x / 2,
                             position_scale * self.cy - position_scale * self.grid_size_y / 2),
                            position_scale * self.r, color='blue')
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.add_artist(circle)
        d = 2
        plt.streamplot(position_scale * x, position_scale * y, w, v, density=d, linewidth=1 / d, arrowsize=1 / d)
        ax.set_aspect('equal')
        colors = self.generate_rainbow_color_scheme(len(position))

#画多个轨迹，不标注其他的信息，嵌套两个list，最里面是三维的ndarray
        if isinstance(position[0], list) and (args is None):
            for i in range(len(position)):
                position[i] = [position_scale * position[i][j] for j in range(len(position[i]))]
                color = mcolors.rgb2hex(colors[i])  # 将RGB值转换为十六进制颜色字符串
                ax.plot(np.array(position[i])[:-10, 0], np.array(position[i])[:-10, 1]
                        , color=color)
                plt.scatter(position[i][0][0], position[i][0][1], c=color)

            # plt.scatter(args["start_point"][0], args["start_point"][1], c="r",label="Source Point")
            # plt.scatter(args["goal_point"][0], args["goal_point"][1], c="g",label="Destination Point")
            plt.xlabel('X(m)')
            plt.ylabel('Y(m)')
            plt.savefig("./draw/fig/thruster_off.png", bbox_inches='tight', dpi=200)

#画两个算法的对比图，因此需要传入很多的args
        elif (args is not None and args.get("multi", False)):
            labels = args["label"]
            annotated = args["annotated"]
            for i in range(len(position)):
                position[i] = [position_scale * position[i][j] for j in range(len(position[i]))]
                color = mcolors.rgb2hex(colors[i])  # 将RGB值转换为十六进制颜色字符串
                ax.plot(np.array(position[i])[:-10, 0], np.array(position[i])[:-10, 1]
                        , color=color, label=labels[i])
                plt.scatter(position[i][0][0], position[i][0][1], c=color)
                half_len = int(len(position[i]) / 2)
                plt.annotate(annotated[i], (position[i][half_len][0], position[i][half_len][1]),
                             textcoords="offset points", xytext=((-1) ** i * 60, -75), ha='center', fontsize=14,
                             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.1'))
            # plt.text(args["start_point"][0] - 50, args["start_point"][1] - 25, "Source Point",
            #          fontsize=18,)  # 添加起始点标记及文字说明
            # plt.text(args["goal_point"][0] - 75, args["goal_point"][1] + 25, "Destination Point",
            #          fontsize=16)  # 添加起始点标记及文字说明
            plt.scatter(args["start_point"][0], args["start_point"][1], c="r",label="Source Point")
            plt.scatter(args["goal_point"][0], args["goal_point"][1], c="g",label="Destination Point")
            plt.xlabel('X(m)')
            plt.ylabel('Y(m)')
            plt.legend()

        # 按照推进器大小设置颜色
        elif (args is not None and args.get("cmap") is not None):
            # ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], c=args.get("cmap_value", "orange")/1000.,
            #         cmap=cm.get_cmap('jet'))
            if True:
                position = [position_scale * position[j] for j in range(len(position))]

                # cmap = cm.get_cmap('jet')  viridis coolwarm cool RdYlBu  Set1 Set2 Pastel
                cmap = cm.get_cmap('cool')  # 使用预定义的颜色映射，例如'jet'
                scatter = ax.scatter(np.array(position)[:, 0], np.array(position)[:, 1],
                                     c=args.get("cmap_value", "orange") / 1000., cmap=cmap,
                                     sizes=2*np.ones(len(position))
                                     )  # 使用速度作为颜色，viridis是一个预定义的颜色映射
                # ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], 'k--')  # 绘制轨迹线，使用黑色虚线
                # ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], 'k--', c=cmap(args.get("cmap_value", "orange")/1000.))
                # 设置颜色条
                cbar = plt.colorbar(scatter)
                cbar.set_label('Thruster Speed')
                plt.scatter(self.start_point[0], self.start_point[1], c="r")
                plt.scatter(self.goal_point[0], self.goal_point[1], c="g")
                try:
                    plt.savefig(f"./draw/fig/color/推进器转速1.png", bbox_inches='tight', dpi=200)
                except:
                    plt.savefig(f"../fig/color/推进器转速1.png", bbox_inches='tight', dpi=200)

                plt.show()
            else:
                position = [position_scale * position[j] for j in range(len(position))]
                color_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
                              'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
                              'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired',
                              'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
                              'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
                              'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
                              'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                              'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
                              'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
                              'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
                              'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
                              'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
                              'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
                              'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
                              'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                              'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',
                              'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic',
                              'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20',
                              'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo',
                              'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
                              'viridis_r', 'winter', 'winter_r']
                for color in color_list:
                    circle = plt.Circle((position_scale * self.cx - position_scale * self.grid_size_x / 2,
                                         position_scale * self.cy - position_scale * self.grid_size_y / 2),
                                        position_scale * self.r, color='blue')
                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    ax.add_artist(circle)
                    d = 2
                    plt.streamplot(position_scale * x, position_scale * y, w, v, density=d, linewidth=1 / d,
                                   arrowsize=1 / d)
                    ax.set_aspect('equal')


                    # cmap = cm.get_cmap('jet')  viridis coolwarm cool RdYlBu  Set1 Set2 Pastel
                    cmap = cm.get_cmap(color)  # 使用预定义的颜色映射，例如'jet'
                    ax.scatter(np.array(position)[:, 0], np.array(position)[:, 1],
                                         c=args.get("cmap_value", "orange") / 1000.,
                                         cmap=cmap)  # 使用速度作为颜色，viridis是一个预定义的颜色映射
                    ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], 'k--')  # 绘制轨迹线，使用黑色虚线
                    # ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], 'k--', c=cmap(args.get("cmap_value", "orange")/1000.))
                    # 设置颜色条
                    # cbar = plt.colorbar(scatter)
                    # cbar.set_label('Thruster Speed')
                    plt.scatter(self.start_point[0], self.start_point[1], c="r")
                    plt.scatter(self.goal_point[0], self.goal_point[1], c="g")
                    plt.savefig(f"./draw/fig/color/{color}.png", bbox_inches='tight', dpi=200)
                    # plt.show()
                    # plt.clf()
                    # plt.cla()
        elif (args is not None and args.get("start_location") is not None):
            position = [position_scale * position[j] for j in range(len(position))]

            ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], c="orange")
            plt.scatter(position_scale * args.get("start_location")[0], position_scale * args.get("start_location")[1], c="r")
            plt.scatter(position_scale * args.get("goal_location")[0], position_scale * args.get("goal_location")[1], c="g")
        else:#普普通通一幅图
            ax.plot(np.array(position)[:, 0], np.array(position)[:, 1], c="orange")
            plt.scatter(self.start_point[0], self.start_point[1], c="r")
            plt.scatter(self.goal_point[0], self.goal_point[1], c="g")

        if prefix is None:
            plt.show()
        else:
            plt.savefig(prefix, bbox_inches='tight', dpi=200)
        plt.cla()
        plt.clf()
        plt.close()

#绘制Velocity Magnitude图片
        if False:
            fig, ax = plt.subplots()
            # ax.plot(np.array(positon)[:,0], np.array(positon)[:,1])
            im = ax.imshow(np.sqrt(self.u[0] ** 2 + self.u[1] ** 2).transpose(),
                           extent=[-260, 260, -180, 180],
                           cmap=cm.Blues)  # 5 Purples, Re26 YlGn, Re65 Blues, Re220 Reds.
            ax.set_xlabel('X(m)')
            ax.set_ylabel('Y(m)')
            ax.set_xlim(-260, 260)
            # ax.set_title('Position and Velocity Field')
            cbar = fig.colorbar(im, shrink=0.7)
            cbar.set_label('Velocity Magnitude')
            # ax.legend()
            # print(Iter)
            # plt.savefig("vel."+str(Iter/100).zfill(4)+".png", bbox_inches='tight', dpi=200)
            plt.savefig("./draw/fig/Velocity Magnitude.png", bbox_inches='tight', dpi=200)
            plt.show()
            haha = True

    def generate_current(self, input_x, input_y, input_z, t):
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

    def generate_rainbow_color_scheme(self, color_count):

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
            hue = int(200 + (i * step) / 360 * 100)
            # 将色相转换为RGB值
            rgb = hsv_to_rgb(hue / 360, 1, 1)
            # 将RGB值添加到颜色列表中
            colors.append(rgb)

        return colors

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(200, 300, 10):
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
