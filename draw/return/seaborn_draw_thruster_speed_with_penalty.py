# 导入库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

def exponential_moving_average(data, alpha=0.8):
    ema = np.zeros_like(data)  # 创建一个与输入数组相同形状的全零数组
    ema[0] = data[0]  # 将第一个数据点设置为初始值

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]  # 计算指数移动平均值

    return ema
# plt.style.use(['science', 'ieee'])

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data1 = pd.read_csv('../plot_demo_files/thruster_speed_n/thruster_speed_n_0.1.csv')
x1 = data1.iloc[:, 0].values
df1 = pd.DataFrame({r'Iterations$(10^6)$': x1, "thruster speed $n$": exponential_moving_average(data1.iloc[:, 4].values)})
df1['parameter'] = r"$\delta$=0.1"

data1 = pd.read_csv('../plot_demo_files/thruster_speed_n/thruster_speed_n_0.5.csv')
x1 = data1.iloc[:, 0].values
df2 = pd.DataFrame({r'Iterations$(10^6)$': x1, "thruster speed $n$": exponential_moving_average(data1.iloc[:, 4].values)})
df2['parameter'] = r"$ \delta $=0.5"

data1 = pd.read_csv('../plot_demo_files/thruster_speed_n/thruster_speed_n_1.0.csv')
x1 = data1.iloc[:, 0].values
df3 = pd.DataFrame({r'Iterations$(10^6)$': x1, "thruster speed $n$": exponential_moving_average(data1.iloc[:, 4].values)})
df3['parameter'] = r"$ \delta $=1.0"

df = df1.append(df2).append(df3)
# df = df2

# df.index = range(len(df))
# print(df)
# 设置图片大小
plt.figure(figsize=(6, 4))
# 画图
# df = df_
sns.lineplot(data=df, x=r'Iterations$(10^6)$', y="thruster speed $n$", hue="parameter",estimator='mean')
# sns.lineplot(data=df_, x=r'Iterations$(10^6)$', y="Episode Return", hue="algo",alpha=0.8)
# sns.lineplot(data=df_, x=r'Iterations$(10^6)$', y="Episode Return",hue="algo2")
# plt.savefig('thruster_speed_with_penalty.png', dpi=300, bbox_inches='tight')
plt.show()
haha=True












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
