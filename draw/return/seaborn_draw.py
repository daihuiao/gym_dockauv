# 导入库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

def exponential_moving_average(data, alpha=0.2):
    ema = np.zeros_like(data)  # 创建一个与输入数组相同形状的全零数组
    ema[0] = data[0]  # 将第一个数据点设置为初始值

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]  # 计算指数移动平均值

    return ema
# plt.style.use(['science', 'ieee'])

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data1 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.9_seed0.csv')[::10]
data2 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.9_seed1.csv')[::10]
data3 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.9_seed4.csv')[::10]
x1 = (data1.iloc[:, 0].values - 1)
x2 = (data2.iloc[:, 0].values - 1)
x3 = (data3.iloc[:, 0].values - 1)
min_x = min(len(x1), len(x2), len(x3))
np_x = np.array(range(min_x))
df1 = pd.DataFrame({r'Iterations$(10^6)$': np_x, "Episode Return": exponential_moving_average(data1.iloc[:, 3][:min_x].values)})
df2 = pd.DataFrame({r'Iterations$(10^6)$': np_x, "Episode Return": exponential_moving_average(data2.iloc[:, 3][:min_x].values)})
df3 = pd.DataFrame({r'Iterations$(10^6)$': np_x, "Episode Return": exponential_moving_average(data3.iloc[:, 3][:min_x].values)})
# df = df1
df = df1.append(df2.append(df3))
df['algo'] = r"$\tau$=0.9"

data1_ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.5_seed0.csv')[::10]
data2_ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.5_seed1.csv')[::10]
data3_ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.5_seed4.csv')[::10]
x1_ = (data1_.iloc[:, 0].values - 1)
x2_ = (data2_.iloc[:, 0].values - 1)
x3_ = (data3_.iloc[:, 0].values - 1)
min_x_ = min(len(x1_), len(x2_), len(x3_))
np_x_ = np.array(range(min_x_))
df1_ = pd.DataFrame({r'Iterations$(10^6)$': np_x_, "Episode Return": exponential_moving_average(data1_.iloc[:, 3][:min_x_].values)})
df2_ = pd.DataFrame({r'Iterations$(10^6)$': np_x_, "Episode Return": exponential_moving_average(data2_.iloc[:, 3][:min_x_].values)})
df3_ = pd.DataFrame({r'Iterations$(10^6)$': np_x_, "Episode Return": exponential_moving_average(data3_.iloc[:, 3][:min_x_].values)})
df_ = df1_.append(df2_.append(df3_))
df_['algo'] = r"$ \tau $=0.5"

data1__ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed0.csv')[::10]
data2__ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed1.csv')[::10]
data3__ = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed4.csv')[::10]
x1__ = (data1__.iloc[:, 0].values - 1)
x2__ = (data2__.iloc[:, 0].values - 1)
x3__ = (data3__.iloc[:, 0].values - 1)
min_x__ = min(len(x1__), len(x2__), len(x3__))
np_x__ = np.array(range(min_x__))
df1__ = pd.DataFrame({r'Iterations$(10^6)$': np_x__, "Episode Return": exponential_moving_average(data1__.iloc[:, 3][:min_x__].values)})
df2__ = pd.DataFrame({r'Iterations$(10^6)$': np_x__, "Episode Return": exponential_moving_average(data2__.iloc[:, 3][:min_x__].values)})
df3__ = pd.DataFrame({r'Iterations$(10^6)$': np_x__, "Episode Return": exponential_moving_average(data3__.iloc[:, 3][:min_x__].values)})
df__ = df1__.append(df2__.append(df3__))
df__['algo'] = r"$ \tau $=0.1"

# 重新排列索引
df = df.append(df_)
df = df.append(df__)
# df.index = range(len(df))
# print(df)
# 设置图片大小
plt.figure(figsize=(6, 4))
# 画图
# df = df_
sns.lineplot(data=df, x=r'Iterations$(10^6)$', y="Episode Return", hue="algo",estimator='mean')
# sns.lineplot(data=df_, x=r'Iterations$(10^6)$', y="Episode Return", hue="algo",alpha=0.8)
# sns.lineplot(data=df_, x=r'Iterations$(10^6)$', y="Episode Return",hue="algo2")
# plt.savefig('image6_.png', dpi=300, bbox_inches='tight')
plt.show()












# # 导入库函数
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scienceplots
#
# # plt.style.use(['science', 'ieee'])
#
# plt.style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# data1 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed0.csv')
# data2 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed1.csv')
# data3 = pd.read_csv('../plot_demo_files/2024_demo1/tau0.1_seed4.csv')
# # x = (data1.iloc[:, 0].values - 1) * 10 ** -6
# x1 = (data1.iloc[:, 0].values - 1) * 10 ** -6
# x2 = (data2.iloc[:, 0].values - 1) * 10 ** -6
# x3 = (data3.iloc[:, 0].values - 1) * 10 ** -6
# min_len = min(len(x1), len(x2), len(x3))
# df1 = pd.DataFrame({r'Episode$(10^6)$': x1[0:min_len], "Episode Return": data1.iloc[:, 4][0:min_len].values})
# df2 = pd.DataFrame({r'Episode$(10^6)$': x2[0:min_len], "Episode Return": data2.iloc[:, 4][0:min_len].values})
# df3 = pd.DataFrame({r'Episode$(10^6)$': x3[0:min_len], "Episode Return": data3.iloc[:, 4][0:min_len].values})
#
# # x = (data1.iloc[:, 0].values - 1) * 10 ** -6
#
# # x = (data1.iloc[:, 0].values - 1) * 10 ** -6
#
# # df2 = pd.DataFrame({r'Iterations$(10^6)$': x, "Episode Return": data1.iloc[:, 4].values})
# # df3 = pd.DataFrame({r'Iterations$(10^6)$': x, "Episode Return": data1.iloc[:, 7].values})
# df  = df1
# # df = df1.append(df2.append(df3))
# for i in range(3):
#     df['algo'] = "FDQL"
#
# # data2 = pd.read_csv('6_imitation_learning.csv')
# # x = (data1.iloc[:, 0].values - 1) * 10 ** -6
# # df1_ = pd.DataFrame({r'Iterations$(10^6)$': x, "Episode Return": data2.iloc[:, 1].values})
# # df2_ = pd.DataFrame({r'Iterations$(10^6)$': x, "Episode Return": data2.iloc[:, 4].values})
# # df3_ = pd.DataFrame({r'Iterations$(10^6)$': x, "Episode Return": data2.iloc[:, 7].values})
# # df_ = df1_.append(df2_.append(df3_))
# # for i in range(3):
# #     df_['algo'] = "imitation learning"
# # 重新排列索引
# df.index = range(len(df))
# print(df)
# # 设置图片大小
# plt.figure(figsize=(6, 4))
# # 画图
# # df=df.append(df_)
# sns.lineplot(data=df, x=r'Iterations$(10^6)$', y="Episode Return", hue="algo")
# # sns.lineplot(data=df_, x=r'Iterations$(10^6)$', y="Episode Return",hue="algo2")
# # plt.savefig('image6_.png', dpi=300, bbox_inches='tight')
# plt.show()
