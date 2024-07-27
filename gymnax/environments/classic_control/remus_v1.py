"""JAX compatible version of CartPole-v1 OpenAI gym environment."""
import pathlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

import copy
import datetime
import importlib
import logging
import os
import pprint
import time
from abc import abstractmethod
from timeit import default_timer as timer
from typing import Tuple, Optional, Union
import numpy as np

import gym
from gym.spaces import Discrete, Box
# import gymnasium as gym
# from gymnasium.spaces import Box
import matplotlib.pyplot as plt
# import numpy asjnp
import jax.numpy as jnp
import wandb
from gym.utils import seeding
from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.objects.current import Current
from gym_dockauv.objects.sensor import Radar
from gym_dockauv.objects.auvsim import AUVSim
from gym_dockauv.objects.shape import Sphere, Spheres, Capsule, intersec_dist_line_capsule_vectorized, \
    intersec_dist_lines_spheres_vectorized, collision_sphere_spheres, collision_capsule_sphere
import gym_dockauv.objects.shape as shape
import gym_dockauv.utils.geomutils as geom
import pickle

logger = logging.getLogger(__name__)

from .remus100_env_base1 import remus100_base1

def get_argument():
    from gym_dockauv.config.env_config import PREDICT_CONFIG, MANUAL_CONFIG, TRAIN_CONFIG, REGISTRATION_DICT, \
        TRAIN_CONFIG_remus_Karman

    used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
    used_TRAIN_CONFIG["vehicle"] = "remus100"
    start_point = [-0, -10, 0]
    goal_point = [-0, 10, 0]
    used_TRAIN_CONFIG["start_point"] = start_point
    used_TRAIN_CONFIG["goal_point"] = goal_point
    used_TRAIN_CONFIG["bounding_box"] = [26, 18, 20]
    # 计算二范数
    used_TRAIN_CONFIG["max_dist_from_goal"] = jnp.linalg.norm(jnp.array(goal_point) - jnp.array(start_point))
    used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.05 * jnp.linalg.norm(jnp.array(goal_point) - jnp.array(start_point))
    used_TRAIN_CONFIG["max_timesteps"] = 999
    used_TRAIN_CONFIG["title"] = "Training Run"

    @dataclass
    class args:
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        current_on: bool = True
        tau: float = 0.5
        w_velocity: float = 0.1
        thruster_penalty: float = 1.0
        thruster: float = 400
        thruster_min: float = 0.

    used_TRAIN_CONFIG["current_on"] = args.current_on
    used_TRAIN_CONFIG["reward_factors"]["w_velocity"] = args.w_velocity
    used_TRAIN_CONFIG["reward_factors"]["thruster_penalty"] = args.thruster_penalty
    used_TRAIN_CONFIG["thruster"] = args.thruster
    used_TRAIN_CONFIG["thruster_min"] = args.thruster_min
    return used_TRAIN_CONFIG
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

        project_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.__str__()

        # with open("/home/ps/dai/overall/togithub/gym_dockauv/karmen_520_180.pkl", "rb") as f:
        with open(project_path + "/gym_dockauv/current/karmen_520_360.pkl", "rb") as f:
            self.Us = pickle.load(f)
            # Create streamline figure
        # self.u = 10 * copy.deepcopy(self.Us[current_index])
        self.u = 10 * jnp.array(self.Us[current_index])
        self.U = self.u[0, :, :].transpose()
        self.V = self.u[1, :, :].transpose()

        self.xx = jnp.linspace(-range_x, range_x, self.U.shape[1])  #
        self.yy = jnp.linspace(-range_y, range_y, self.U.shape[0])
        self.zz = jnp.linspace(-range_z, range_z, self.grid_size_z)
        self.XX, self.YY, self.ZZ = jnp.meshgrid(self.xx, self.yy, self.zz)

        # Coordinates of the circular obstacle
        self.cx = 2 * range_x / 4
        self.cy = 2 * range_y / 2
        self.r = 2 * range_y / 9

    def validate_input(self, input_val, min_val, max_val, name):
        def true_fun(_):
            # jax.debug.print(f"{name}error ******** error *************** error ************************* ")
            return jnp.array(1.0)

        def false_fun(_):
            return jnp.array(0.0)

        haha1 = lax.cond(input_val < min_val, true_fun, false_fun, operand=None)
        haha2 = lax.cond(input_val > max_val, true_fun, false_fun, operand=None)

    @partial(jax.jit, static_argnums=(0))
    def generate_current(self, input_x, input_y, input_z, t):
        # dai 这一块没有办法验证了，只能保证他在范围内
        # try:
        #     assert bool(float(input_x) > float(self.xx.min()))
        #     assert bool(float(input_x) < float(self.xx.max()))
        #     assert bool(float(input_y) > float(self.yy.min()))
        #     assert bool(float(input_y) < float(self.yy.max()))
        #     assert bool(float(input_z) > float(self.zz.min()))
        #     assert bool(float(input_z) < float(self.zz.max()))
        # except:
        #     print("输入的经纬度不在范围内") # dai 测试这个输入输出 以及条件判断
        #     print("_lon:", input_x, "lon.min:", self.xx.min(), "lon.max:", self.xx.max())
        #     print("_lat:", input_y, "lat.min:", self.yy.min(), "lat.max:", self.yy.max())
        #     print("_alt:", input_z, "alt.min:", self.zz.min(), "alt.max:", self.zz.max())

        # 输入验证
        # jax.debug.print("haha{}", self.xx.min())
        # jax.debug.print("haha{}", input_x)
        # 不好使阿,这个 lax.cond 不管条件是真是假,直接进第一个函数,卧槽
        # self.validate_input(input_x, self.xx.min(), self.xx.max(), "经度")
        # self.validate_input(input_y, self.yy.min(), self.yy.max(), "纬度")
        # self.validate_input(input_z, self.zz.min(), self.zz.max(), "高度")

        ind_x = jnp.sum(input_x >= self.xx) - 1
        ind_y = jnp.sum(input_y >= self.yy) - 1
        u = self.U[ind_y, ind_x]
        v = self.V[ind_y, ind_x]
        w = 0

        return jnp.array([u, v, w])


# todo dai 事变，以后在做吧
# @partial(jax.jit, static_argnums=(0))
# def generate_current_with_t(self, input_x, input_y, input_z,t):
#     try:
#         assert bool(float(input_x) > float(self.xx.min()))
#         assert bool(float(input_x) < float(self.xx.max()))
#         assert bool(float(input_y) > float(self.yy.min()))
#         assert bool(float(input_y) < float(self.yy.max()))
#         assert bool(float(input_z) > float(self.zz.min()))
#         assert bool(float(input_z) < float(self.zz.max()))
#     except:
#         print("输入的经纬度不在范围内")
#
#         print("_lon:", input_x, "lon.min:", self.xx.min(), "lon.max:", self.xx.max())
#         print("_lat:", input_y, "lat.min:", self.yy.min(), "lat.max:", self.yy.max())
#         print("_alt:", input_z, "alt.min:", self.zz.min(), "alt.max:", self.zz.max())
#
#     ind_x = sum(input_x >= self.xx) - 1
#     ind_y = sum(input_y >= self.yy) - 1
#
#     U = 10 * self.Us[400+t//10][0, :, :].transpose()
#     V = 10 * self.Us[400+t//10][1, :, :].transpose()
#
#     u = U[ind_y, ind_x]
#     v = V[ind_y, ind_x]
#     w = 0
#
#     return jnp.array([u, v, w])

class Current:
    r"""
    Ocean current with constant alpha and beta and first order gauss markov process (linear state model) for simulation.

    `Fossen2011 <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>`_, Chapter 8

    :param mu: Constant in
    :param V_min: Lower boundary for current speed
    :param V_max: upper boundary for current speed
    :param Vc_init: Initial current speed
    :param alpha_init: Initial :math:`\alpha` angle in rad
    :param beta_init: Initial :math:`\beta` angle in rad
    :param white_noise_std: Standard deviation :math:`\sigma` of the white noise
    """

    # self.current = Current(mu=0.005, V_min=0.0, V_max=0.0, Vc_init=0.0,
    #                        alpha_init=jnp.pi / 4, beta_init=jnp.pi / 4, white_noise_std=0.0,
    #                        step_size=self.auv.step_size, current_on=True)
    def __init__(self, karmen_current: Karmen_current,
                 mu: float = 0.005, V_min: float = 0.0, V_max: float = 0.0, Vc_init: float = 0.0,
                 alpha_init: float = jnp.pi / 4, beta_init: float = jnp.pi / 4,
                 white_noise_std: float = 0.0, step_size: float = 1,
                 current_on: bool = True, current_scale: float = 1.0,
                 ):
        self.current_on = current_on
        self.mu = mu
        self.V_min = V_min
        self.V_max = V_max
        self.V_c = Vc_init
        self.alpha = alpha_init
        self.beta = beta_init
        self.white_noise_std = white_noise_std
        self.step_size = step_size
        self.current_scale = current_scale
        self.karmen_current = karmen_current

    @partial(jax.jit, static_argnums=(0))
    def __call__(self, Theta: jnp.ndarray, position=None, return_None=False) -> jnp.ndarray:
        phi = Theta[0]
        theta = Theta[1]
        psi = Theta[2]

        vel_current_NED = self.current_scale * self.karmen_current.generate_current(position[0], position[1],
                                                                                    position[2], 0)

        vel_current_BODY = jnp.transpose(geom.Rzyx(phi, theta, psi)).dot(vel_current_NED)

        nu_c = jnp.array([*vel_current_BODY, 0, 0, 0])

        return nu_c

    # def __call__(self, Theta: jnp.ndarray, position=None, return_None=False) -> jnp.ndarray:
    #     r"""
    #     Returns the current velocity :math:`\boldsymbol{\nu}_c` in {b} for the AUV when called
    #
    #     .. math ::
    #
    #         \boldsymbol{v}_c^b = \boldsymbol{R}_b^n(\boldsymbol{\Theta_{nb}})^T \boldsymbol{v}_c^n
    #
    #     :param Theta: Euler Angles array 3x1 :math:`[\phi, \theta, \psi]^T`
    #     :return: 6x1 array
    #     """
    #     phi = Theta[0]
    #     theta = Theta[1]
    #     psi = Theta[2]
    #
    #     if position is None:
    #         raise NotImplementedError
    #         vel_current_NED = self.get_current_NED()
    #         vel_current_BODY = jnp.transpose(geom.Rzyx(phi, theta, psi)).dot(vel_current_NED)
    #
    #         nu_c = jnp.array([*vel_current_BODY, 0, 0, 0])
    #     else:
    #
    #         # vel_current_NED = self.get_current_NED()
    #         if self.current_on:
    #             vel_current_NED = self.current_scale * self.karmen_current.generate_current(position[0], position[1],
    #                                                                                         position[2], 0)
    #         else:
    #             raise NotImplementedError
    #             vel_current_NED = jnp.array([0, 0, 0])
    #         vel_current_BODY = jnp.transpose(geom.Rzyx(phi, theta, psi)).dot(vel_current_NED)
    #
    #         nu_c = jnp.array([*vel_current_BODY, 0, 0, 0])
    #     # if return_None:
    #     #     returnjnp.array([0,0,0,0,0,0])
    #     return nu_c

    def get_current_NED(self) -> jnp.ndarray:
        r"""
        Returns current in NED coordinates

        .. note:: The :math:`\alpha` and :math:`\beta` angles from initialization are assumed to be constant in NED,
            thus transformation from FLOW coordinate system to NED can be done here with varying :math:`V_c`

        .. math::

            \boldsymbol{v}_c = V_c \begin{bmatrix}
                \cos \alpha_c \\
                \sin \beta_c \\
                \sin \alpha_c cos \beta_c
            \end{bmatrix}^T

        :return: 3x1 array
        """
        raise NotImplementedError
        vel_current_NED = jnp.array([self.V_c * jnp.cos(self.alpha) * jnp.cos(self.beta),
                                     self.V_c * jnp.sin(self.beta),
                                     self.V_c * jnp.sin(self.alpha) * jnp.cos(self.beta)])

        return vel_current_NED

    def sim(self, rng, V_c):  # todo dai 这个函数还有问题，直接查表的肯定没问题了，但是自己计算的这个还是一幅面向对象的样子，Vc需要及时输入
        r"""
        Simulate one time step of the current dynamics according to linear state model

        .. math::

            \dot{V}_c + \mu V_c = w

        :return: None
        """
        raise NotImplementedError
        w = jax.random.normal(rng) * self.white_noise_std
        # w = jnp.random.normal(0, self.white_noise_std)
        Vc_dot = -self.mu * self.V_c + w
        V_c += Vc_dot * self.step_size

        # From Simen
        # V_c = self.V_c+Vc_dot*h
        # self.V_c = 0.99*self.V_c + 0.01*V_c

        V_c = jnp.clip(self.V_c, self.V_min, self.V_max)
        return V_c


class Reward:
    """
    Static class to group the different reward functions
    """

    @staticmethod
    @jax.jit
    def log_precision(x: float, x_goal: float, x_max: float) -> float:
        """
        Function to scale logarithmic function between x_goal->y=0 and x_max->y=1 (clip in case max or des value got
        over-/ or undershot

        :param x: actual value
        :param x_max: maximum value
        :param x_goal: goal value (smaller than max!)
        :return: x value scaled on log (0 <= x <= 1 when x_goal <= x <= x_max)
        """
        """
        GPT
这个 log_precision 函数使用对数函数来将输入值 x 映射到一个从 0 到 1 的范围。它设计用于在两个特定点之间创建一个平滑且非线性的映射。
这个映射在 x_goal 处达到 0，在 x_max 处达到 1。在这个范围之外的值被剪辑到 0 或 1。

参数说明：
x：实际值，是函数的输入。
x_goal：目标值。这是映射中的一个关键点，在这一点上函数的输出为 0。
x_max：最大值。这是映射中的另一个关键点，在这一点上函数的输出为 1。
函数工作原理：
防止对数运算中的零值：

epsilon = 0.001 是一个小量，用来防止对数运算中出现零值，因为 log(0) 是未定义的。
对数映射：

函数首先计算 x 和 x_goal 相对于 x_max 的对数比率。这个比率随着 x 接近 x_max 而增加。
反转和剪辑：

映射是反向的，即随着 x 接近 x_goal，函数值接近 1，而随着 x 接近 x_max，函数值减小到 0。
使用 jnp.clip 函数将结果限制在 0 到 1 之间，以确保即使 x 超出 x_goal 和 x_max 的范围，输出也在这个区间内。
应用场景：
这种类型的函数在需要非线性映射的场景中非常有用，特别是在需要平滑过渡或者将一组数据标准化到一定范围内时。
例如，在控制系统、机器学习模型的评分机制，或者在处理与目标值相关的误差时，这样的映射可以提供更细致的控制或评估机制。
"""
        epsilon = 0.001  # Protection against log(0.0)
        x = jax.lax.select(x < epsilon, epsilon, x)
        x_goal = jax.lax.select(x_goal < epsilon, epsilon, x_goal)

        log_x_ratio = jnp.log(x / x_max)
        log_x_goal_ratio = jnp.log(x_goal / x_max)

        return 1 - jnp.clip(log_x_ratio / log_x_goal_ratio, 0, 1)
        # return 1 - jnp.clip((jnp.log(max(x, epsilon) / x_max) / jnp.log(max(x_goal, epsilon) / x_max)), 0, 1)

    @staticmethod
    @jax.jit
    def disc_goal_constraints(x: float, x_des: float, perc: float = 0.2) -> float:
        """
        Function for the final discrete reward when goal is reached for the constraints. This assumes a desired small
        positive value as a target (e.g. tolerance) and the actual achieves value (positive). Overshoot percentage is
        used to make sure, the final result is always below the desired value.

        :param x: actual value
        :param x_des: desired value
        :param perc: percentage to overshoot desired value
        :return: reward [0..2] for actual value from desired value [0..1]
            extra reward when des value actually reached [0,1]
        """
        x_des -= x_des * perc

        # 使用 jax.lax.select 来处理条件分支
        max_value = jax.lax.select(x > x_des, x, x_des)
        reward = (x_des / max_value) ** 2 + (x < x_des)

        return reward
        # return (x_des / max(x_des, x)) ** 2 + (x < x_des)
        # return (x_des / jnp.max(x_des, x)) + (x < x_des)

    @staticmethod
    @jax.jit
    def cont_goal_constraints(x: float, delta_d: float, x_des: float, delta_d_des: float, x_max: float,
                              delta_d_max: float, x_exp: float = 1.0, delta_d_exp: float = 1.0, x_rev: bool = False,
                              delta_d_rev: bool = False) -> float:
        """
        Continuous reward functions for some goal constraints put into a from goal distance dependent function

        :param x: actual value
        :param delta_d: distance from goal
        :param x_des: desired value
        :param delta_d_des: desired delta distance (needed to form function)
        :param x_max: max value
        :param delta_d_max: maximum delta distance (needed to form function)
        :param x_exp: exponent for term of x
        :param delta_d_exp: exponent for term of delta_d
        :param x_rev: parameter to reverse direction on x
        :param delta_d_rev: parameter to reverse direction on delta_d
        :return: reward [0..1] for actual value x with delta_d
        """
        r_x = jnp.abs((float(x_rev) - Reward.log_precision(x, x_des, x_max))) ** x_exp
        r_delta_d = jnp.abs(
            (float(delta_d_rev) - Reward.log_precision(delta_d, delta_d_des, delta_d_max))) ** delta_d_exp
        return r_x * r_delta_d

    @staticmethod
    @jax.jit
    def obstacle_avoidance(theta_r: jnp.ndarray, psi_r: jnp.ndarray, d_r: jnp.ndarray, theta_max: float, psi_max: float,
                           d_max: float, gamma_c: float, epsilon_c: float, epsilon_oa: float = 0.01):
        """
        function to calculate the reward for the obstacle avoidance mechanism
        :param theta_r: 1d array of the vertical radar angles
        :param psi_r: 1d array of the horizontal radar angles
        :param d_r: 1d array distance of each radar ray
        :param theta_max: maximum angle vertically
        :param psi_max: maximum angle horizontally
        :param d_max: maximum distance for an array
        :param gamma_c: scaling the closeness values
        :param epsilon_c: minimum obstacle closeness punishment
        :param epsilon_oa: avoiding singularities
        :return:
        """
        beta = Reward.beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa)
        c = Reward.c_oa(d_r, d_max)
        return jnp.sum(beta) / (jnp.maximum((gamma_c * (1 - c)) ** 2, epsilon_c) @ beta) - 1  # Sum via scalar product

    @staticmethod
    @jax.jit
    def beta_oa(theta_r, psi_r, theta_max, psi_max, epsilon_oa):
        return (1 - jnp.abs(theta_r) / theta_max) * (1 - jnp.abs(psi_r) / psi_max) + epsilon_oa

    @staticmethod
    @jax.jit
    def c_oa(d_r, d_max):
        return jnp.clip(1 - d_r / d_max, 0, 1)


@struct.dataclass  # (state, state_dot,  last_state,  u_actual,  nu_c, time)
class EnvState(environment.EnvState):
    state: jnp.ndarray
    state_dot: jnp.ndarray
    last_state: jnp.ndarray
    u_actual: jnp.ndarray
    nu_c: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8


class remus_v1(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible version of remus100_env_v1 OpenAI gym environment.

    """

    # 共享的大矩阵作为类变量
    karmen_current = Karmen_current()

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__()
        # self.obs_shape = (12,)

        self.remus = remus100_base1()

        '''docking3d code'''
        # Basic config for logger
        env_config = get_argument()
        self.config = env_config

        try:
            self.current_on = env_config["current_on"]
        except:
            self.current_on = True
        # self.index = env_config["index"] if env_config["index"] is not None else 0

        self.title = self.config["title"]

        # Joystick控制模式: 适用于有限的推力和控制输入，主要用于防止过快移动。推力系数较低，控制输入矩阵B
        # 是基于直接控制假设的。
        # Direct控制模式: 适用于完整的六自由度控制，每个推力器的推力相同且较高。
        # 控制输入矩阵B由推力变换矩阵和推力系数矩阵相乘得到，输入限制较为均匀。
        control_mode = 'joystick'

        if control_mode == 'joystick':
            self.K_thrust = jnp.array(20)  # Reduced maximum thrust here as from [2] for restricting too fast movement
            # B Matrix calculated from assumption in direct control mode with low level control
            self._B = jnp.array([
                [2.83, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.83, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.436, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.24, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.378]
            ]) * self.K_thrust
            self.u_bound = jnp.array([
                [-10, 10],
                [-10, 10],
                [self.config["thruster_min"], self.config["thruster"]] ])  # attention
            # low = jnp.concatenate((self.u_bound[:, 0][0:-1], jnp.array([self.config["thruster_min"]]))),
            # high = jnp.concatenate((self.u_bound[:, 1][0:-1], jnp.array([self.config["thruster"]]))),

            # from C.-J. Wu and B. Eng, “6-DoF Modelling and Control of a Remotely Operated Vehicle,” p. 39.ff
        elif control_mode == 'direct':
            self.K_thrust = jnp.diag([40, 40, 40, 40, 40, 40, 40, 40])  # since each thruster is the same
            self.T_thrust = jnp.array([
                [0.707, 0.707, -0.707, -0.707, 0, 0, 0, 0],
                [-0.707, 0.707, -0.707, 0.707, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, -1, -1, -1],
                [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                [0.06, 0.06, -0.06, -0.06, 0.120, -0.120, 0.120, -0.120],
                [-0.189, 0.189, 0.189, -0.189, 0, 0, 0, 0]
            ])
            self._B = jnp.dot(self.T_thrust, self.K_thrust)
            self.u_bound = jnp.array([
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1]])
        else:
            raise KeyError("Invalid control mode for BlueROV2 initialization.")

        # # Set step size for vehicle
        # self.auv.step_size = self.config["t_step_size"]

        # Set assumption values for vehicle max velocities from config
        self.u_max = jnp.array(self.config["u_max"])
        self.v_max = jnp.array(self.config["v_max"])
        self.w_max = jnp.array(self.config["w_max"])
        self.p_max = jnp.array(self.config["p_max"])
        self.q_max = jnp.array(self.config["q_max"])
        self.r_max = jnp.array(self.config["r_max"])

        # # Navigation errors
        # self.delta_d = 0
        # self.delta_psi = 0
        # self.delta_psi_ = 0
        # self.delta_theta = 0
        # self.delta_heading_goal = 0  # This is the error for the heading that is required AT goal

        # Water current
        self.current = Current(remus_v1.karmen_current)
        # self.nu_c = self.current(self.auv.attitude, position=self.auv.position)

        # Init radar sensor suite
        self.radar_args = self.config["radar"]

        # self.radar = Radar(eta=self.auv.eta, **self.radar_args)#
        self.radar = Radar(eta=jnp.array([0] * 6), **self.radar_args)  # todo dai to reset

        # Init list of obstacles (that will collide with the vehicle or have intersection with the radar)
        # Keep copy of capsules and spheres, as they are the only one supported so far:
        self.capsules = []  # type: list[Capsule]
        self.spheres = Spheres([])  # type: Spheres
        self.obstacles = [*self.capsules, *self.spheres()]  # type: list[shape.Shape]

        # Set the action and observation space
        self.bounding_box = jnp.array(self.config["bounding_box"])
        self.n_obs_without_radar = 16 + 3
        # self.n_observations = self.n_obs_without_radar + self.radar.n_rays_reduced
        self.n_observations = self.n_obs_without_radar + 0
        self.obs_shape = (self.n_observations,)


        # The inner lists decide, in which subplot the observations will go
        self.meta_data_observation = [
            ["delta_d", "delta_theta", "delta_psi"],
            ["u", "v", "w"],
            ["phi", "theta", "psi_sin", "psi_cos"],
            ["p", "q", "r"],
            ["u_c", "v_c", "w_c"],
            [f"ray_{i}" for i in range(self.radar.n_rays_reduced)]
        ]

        self.max_timesteps = self.config["max_timesteps"]

        # Rewards
        # self.reward_set = self.config["reward_set"]  # Chosen reward set
        self.n_rewards = 13 + 3  # Number helps to structure rewards
        self.n_cont_rewards = 8
        # self.last_reward = 0  # Last reward
        # self.last_reward_arr = jnp.zeros(self.n_rewards)  # This should reflect the dimension of the rewards parts
        # self.cumulative_reward = 0  # Current cumulative reward of agent
        # self.cum_reward_arr = jnp.zeros(self.n_rewards)
        # self.conditions = None  # Boolean array to see which conditions are true
        # Description for the meta data
        self.meta_data_reward = [
            "Nav_delta_d",
            "Nav_delta_theta",
            "Nav_delta_psi",
            "Att_phi",
            "Att_theta",
            # Depracated, Goal constraints removed - Erik - 30.06.2022
            # "pdot",
            "Thetadot",
            # "Goal_psi_g",
            "obstacle_avoid",
            # "time_step",
            "action",
            "Done-Goal_reached",
            "Done-out_pos",
            "Done-out_att",
            "Done-max_t",
            "Done-collision"
        ]
        self.reward_factors = self.config["reward_factors"]  # Dictionary containing weights!
        # Extract done rewards in an array
        self.w_done = jnp.array([
            self.reward_factors["w_goal"],
            self.reward_factors["w_deltad_max"],
            self.reward_factors["w_Theta_max"],
            self.reward_factors["w_t_max"],
            self.reward_factors["w_col"]
        ])

        self.meta_data_done = self.meta_data_reward[self.n_cont_rewards:]
        # self.goal_constraints = []  # List of booleans for further contraints as soon as goal is reached
        self.goal_location = None  # This needs to be defined in self.generate_environment
        self.start_location = None  # This needs to be defined in self.generate_environment
        self.dist_goal_reached_tol = self.config[
            "dist_goal_reached_tol"]  # Distance tolerance for successfully reached goal
        self.velocity_goal_reached_tol = self.config["velocity_goal_reached_tol"]  # Velocity limit at goal
        self.ang_rate_goal_reached_tol = self.config["ang_rate_goal_reached_tol"]  # Angular rate limit at goal
        self.attitude_goal_reached_tol = self.config["attitude_goal_reached_tol"]  # Attitude tolerance at goal
        self.max_dist_from_goal = self.config["max_dist_from_goal"]
        self.max_attitude = self.config["max_attitude"]
        self.heading_goal_reached = 0  # Heading at goal, pitch should be zero

        if "goal_point" in env_config:
            self.goal_point_ = jnp.array(env_config["goal_point"])
        else:
            raise NotImplementedError
        if "start_point" in env_config:
            self.start_point_ = jnp.array(env_config["start_point"])

        self.generate_environment()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    @partial(jax.jit, static_argnums=(0))
    def step_env(
            self,
            key: chex.PRNGKey,
            envState: EnvState,
            action: Union[int, float, chex.Array],
            params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # key, key_use = jax.random.split(key, 2)
        # self.current.sim(key_use,V_c=xxx)
        input_c = jnp.clip(action, -1., 1.)
        action = self.u_bound[:, 0] + (self.u_bound[:, 1] - self.u_bound[:, 0]) * (input_c + 1) / 2

        u = action  # bug fixed ,why? 峨峨峨,我知道了，他原来的代码就有问题，sb3本来就会做一个去皈依化，而他可能之前用的其他算法库。

        last_state = envState.state
        state_dot = envState.state_dot
        last_state_real = envState.last_state
        u_actual = envState.u_actual
        # nu_c = envState.nu_c
        nu_c = self.current(Theta=last_state[3:6], position=last_state[0:3])
        time = envState.time

        # self._sim(nu_c) 下面就是这个函数的内容

        state, u_actual, state_dot = self.remus.remus_solver(u, eta=last_state[:6],
                                                             nu=last_state[6:],
                                                             nu_c=nu_c,
                                                             u_actual=u_actual)

        # Convert angle in applicable range
        state = state.at[3:6].set(geom.ssa(state[3:6]))  # 将欧拉角限制在[-pi, pi]之间

        # Important: Reward is based on termination is previous step transition

        # Update state dict and evaluate termination conditions
        envstate = EnvState(
            state=state,
            state_dot=state_dot,
            last_state=last_state,
            u_actual=u_actual,
            nu_c=nu_c,
            time=envState.time + 1,
        )
        reward = self.reward_step(action, envstate)  # todo dai 奖励函数

        done = self.is_terminal(envstate, params)  # todo dai done

        return (
            lax.stop_gradient(self.get_obs(envstate)),
            lax.stop_gradient(envstate),
            jnp.array(reward),
            done,
            {"discount": self.discount(envstate, params)},
        )

    @partial(jax.jit, static_argnums=(0))
    def reward_step(self, action: jnp.ndarray, envstate: EnvState) -> float:
        """
        Calculate the reward function, make sure to call self.is_done() before to update and check the done conditions

        The factors are defined in the config. Each reward is normalized between 0..1, thus the factor decides its
        importance. Keep in mind the rewards for the done conditions will be sparse.

        Reward 1: Navigation Errors
        Reward 2: Stable attitude
        Reward 3: Goal constraints
        Reward 3: time step penalty
        Reward 4: action use penalty
        Reward 5: Done - Goal reached
        Reward 6: Done - out of bounds position
        Reward 7: Done - out of bounds attitude
        Reward 8: Done - maximum episode steps
        Reward 9: Done - collision

        :param action: array with actions between -1 and 1
        :return: The single reward at this step

        "w_d": 1.1,                         # Continuous: distance from goal
        "w_delta_psi": 0.5,                 # Continuous: chi error (heading) yaw
        "w_delta_theta": 0.3,               # Continuous: delta_theta error (elevation) pitch
        "w_phi": 0.3,                       # Continuous: phi error (roll angle) roll
        "w_theta": 0.3,                     # Continuous: theta error (pitch angle)
        "w_Thetadot": 0.2,                  # Continuous: total angular rate
        "w_t": 0.05,                        # Continuous: constant time step punish
        "w_oa": 0.20,                        # Continuous: obstacle avoidance parameter
        "w_goal": 400.0,                    # Discrete: reaching goal
        "w_deltad_max": -200.0,             # Discrete: Flying out of bounds
        "w_Theta_max": -200.0,              # Discrete: Too high attitude
        "w_t_max": -100.0,                  # Discrete: Episode maximum length over
        "w_col": -300.0,                    # Discrete: Collision factor
        last_reward_arr[0]与目标点有关
        【1】【2】是调整朝向，也和目标点有关
        【3】【4】是约束roll和pitch为0
        【5】是约束角速度为0
        【6】是避障奖励
        [7] 限制动作幅度
        【8】-【13】是结束奖励，到达目标给奖励，其余给惩罚
                Condition 0: Check if close to the goal 400
                Condition 1: Check if out of bounds for position -200
                Condition 2: Check if attitude (pitch, roll) too high
                Condition 3: Check if maximum time steps reached
                Condition 4: Check for collision
        """

        diff = self.goal_location - envstate.state[0:3]
        # self.delta_d 表示的是 AUV 与目标位置之间的距离
        delta_d = jnp.linalg.norm(diff)
        last_reward_arr_0 = -self.reward_factors["w_d"] * Reward.log_precision(
            x=delta_d,
            x_goal=self.dist_goal_reached_tol,
            x_max=self.max_dist_from_goal
        )

        # reward_set 好像一直都是1

        # self.delta_theta 表示的是为了从 AUV 的当前位置和姿态移动到目标位置所需改变的俯仰角.ssa:Smallest Signed Angle
        # delta_theta = self.auv.attitude[1] + (geom.ssa(jnp.arctan2(diff[2], jnp.linalg.norm(diff[:2]))))
        delta_theta = envstate.state[4] + (geom.ssa(jnp.arctan2(diff[2], jnp.linalg.norm(diff[:2]))))
        last_reward_arr_1 = - self.reward_factors["w_delta_theta"] * (
                delta_theta / (jnp.pi / 2)) ** 2  # pitch

        # self.delta_psi 表示 AUV 为了朝向目标位置所需做出的偏航角度
        # self.delta_psi = geom.ssa(jnp.arctan2(diff[1], diff[0]) - self.auv.attitude[2])
        delta_psi = geom.ssa(jnp.arctan2(diff[1], diff[0]) - envstate.state[5])
        last_reward_arr_2 = - self.reward_factors["w_delta_psi"] * (delta_psi / jnp.pi) ** 2  # yaw

        # Reward for stable attitude
        # phi 是roll,合理，因为roll是不需要调整的，所以这个奖励是为了让roll保持稳定
        # self.last_reward_arr[3] = -self.reward_factors["w_phi"] * (self.auv.attitude[0] / (jnp.pi / 2)) ** 2
        last_reward_arr_3 = -self.reward_factors["w_phi"] * (envstate.state[3] / (jnp.pi / 2)) ** 2

        # todo dai: theta 是pitch，这个合理吗，让他一直保持为0？ 0727 让他两次变化尽可能小，合理，记得改
        last_reward_arr_4 = -self.reward_factors["w_theta"] * 0 * (
            # (self.auv.attitude[1] - self.auv.last_attitude[1]) / (jnp.pi / 2)) ** 2
                (envstate.state[4] - envstate.last_state[3]) / (jnp.pi / 2)) ** 2

        last_reward_arr_5 = - self.reward_factors["w_Thetadot"] * (
            # jnp.linalg.norm(self.auv.euler_dot) / self.p_max) ** 2  # 欧拉角是角度，欧拉角的导数是角速度，让角速度尽可能小。
                jnp.linalg.norm(envstate.state_dot[3:6]) / self.p_max) ** 2  # 欧拉角是角度，欧拉角的导数是角速度，让角速度尽可能小。
        # Reward function for obstacle avoidance,

        last_reward_arr_6 = - self.reward_factors["w_oa"] * Reward.obstacle_avoidance(
            theta_r=self.radar.alpha, psi_r=self.radar.beta, d_r=self.radar.intersec_dist,
            theta_max=self.radar.alpha_max, psi_max=self.radar.beta_max, d_max=self.radar.max_dist,
            gamma_c=1, epsilon_c=0.001, epsilon_oa=0.01)

        # if self.reward_set == 1:
        #     # Set 1
        #     # self.delta_theta 表示的是为了从 AUV 的当前位置和姿态移动到目标位置所需改变的俯仰角.ssa:Smallest Signed Angle
        #     self.last_reward_arr[1] = - self.reward_factors["w_delta_theta"] * (
        #             self.delta_theta / (jnp.pi / 2)) ** 2  # pitch
        #     self.last_reward_arr[2] = - self.reward_factors["w_delta_psi"] * (self.delta_psi / jnp.pi) ** 2  # yaw
        # elif self.reward_set == 2:
        #     # Set 2
        #     self.last_reward_arr[1] = - self.reward_factors["w_delta_theta"] * Reward.cont_goal_constraints(
        #         x=jnp.abs(self.delta_theta),
        #         delta_d=self.delta_d,
        #         x_des=0.0,
        #         delta_d_des=self.dist_goal_reached_tol,
        #         x_max=jnp.pi / 2,
        #         delta_d_max=self.max_dist_from_goal,
        #         x_exp=4.0,
        #         delta_d_exp=4.0,
        #         x_rev=False,
        #         delta_d_rev=False
        #     )
        #     self.last_reward_arr[2] = - self.reward_factors["w_delta_psi"] * Reward.cont_goal_constraints(
        #         x=jnp.abs(self.delta_psi),
        #         delta_d=self.delta_d,
        #         x_des=0.0,
        #         delta_d_des=self.dist_goal_reached_tol,
        #         x_max=jnp.pi,
        #         delta_d_max=self.max_dist_from_goal,
        #         x_exp=4.0,
        #         delta_d_exp=4.0,
        #         x_rev=False,
        #         delta_d_rev=False
        #     )
        # if self.reward_set == 1:
        #     # Set 1
        #     self.last_reward_arr[6] = - self.reward_factors["w_oa"] * Reward.obstacle_avoidance(
        #         theta_r=self.radar.alpha, psi_r=self.radar.beta, d_r=self.radar.intersec_dist,
        #         theta_max=self.radar.alpha_max, psi_max=self.radar.beta_max, d_max=self.radar.max_dist,
        #         gamma_c=1, epsilon_c=0.001, epsilon_oa=0.01)
        # elif self.reward_set == 2:
        #     # Set 2
        #     roa = Reward.obstacle_avoidance(
        #         theta_r=self.radar.alpha, psi_r=self.radar.beta, d_r=self.radar.intersec_dist,
        #         theta_max=self.radar.alpha_max, psi_max=self.radar.beta_max, d_max=self.radar.max_dist,
        #         gamma_c=1, epsilon_c=0.001, epsilon_oa=0.01)
        #
        #     self.last_reward_arr[6] = - self.reward_factors["w_oa"] * Reward.cont_goal_constraints(
        #         x=jnp.abs(roa),
        #         delta_d=self.delta_d,
        #         x_des=0.0,
        #         delta_d_des=self.dist_goal_reached_tol,
        #         x_max=1,
        #         delta_d_max=self.max_dist_from_goal,
        #         x_exp=4.0,
        #         delta_d_exp=4.0,
        #         x_rev=False,
        #         delta_d_rev=False
        #     )

        last_reward_arr_7 = - (jnp.sum(
            (jnp.abs(action) / self.u_bound.shape[0]) ** 2 * self.reward_factors["action_reward_factors"]))

        # Add extra reward on checking which condition caused the episode to be done (discrete rewards)
        # conditions = [
        #     # Condition 0: Check if close to the goal
        #     delta_d < self.dist_goal_reached_tol,
        #     # Condition 1: Check if out of bounds for position
        #     # self.delta_d > self.max_dist_from_goal,
        #     jnp.abs(envstate.state[0]) > self.bounding_box[0] or
        #     abs(envstate.state[1]) > self.bounding_box[1] or
        #     abs(envstate.state[2]) > self.bounding_box[2],
        #     # Condition 2: Check if attitude (pitch, roll) too high
        #     # np.any(np.abs(self.auv.attitude[:2]) > self.max_attitude),
        #     False,
        #     # Condition 3: Check if maximum time steps reached
        #     envstate.time >= self.max_timesteps,
        #     # Condition 4: Collision with obstacle (is updated earlier)
        #     # self.collision
        #     False
        # ]
        out_of_bounds = jnp.logical_or(
            jnp.logical_or(jnp.abs(envstate.state[0]) > self.bounding_box[0],
                           jnp.abs(envstate.state[1]) > self.bounding_box[1]),
            jnp.abs(envstate.state[2]) > self.bounding_box[2]
        )

        conditions = jnp.array([
            # Condition 0: Check if close to the goal
            delta_d < self.dist_goal_reached_tol,
            # Condition 1: Check if out of bounds for position
            out_of_bounds,
            # Condition 2: Check if attitude (pitch, roll) too high
            # jnp.any(jnp.abs(self.auv.attitude[:2]) > self.max_attitude),
            False,
            # Condition 3: Check if maximum time steps reached
            envstate.time >= self.max_timesteps,
            # Condition 4: Collision with obstacle (is updated earlier)
            # self.collision
            False
        ])

        last_reward_arr_8_12 = jnp.sum(jnp.array(conditions) * self.w_done)

        # Just for analyzing purpose:
        # self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr

        reward = last_reward_arr_0 + last_reward_arr_2 + last_reward_arr_3 + last_reward_arr_4 + last_reward_arr_5 + \
                 last_reward_arr_6 + last_reward_arr_7 + last_reward_arr_8_12

        velocity_reward = self.reward_factors["w_velocity"] * \
                          jnp.linalg.norm(envstate.state_dot[0:2])  # todo dai: 这里想用横向上的速度作为奖励，
        # last_reward_arr_13 = velocity_reward
        # reward += velocity_reward

        # diff = self.goal_location - envstate.state[0:3]
        # delta_d = jnp.linalg.norm(diff)
        last_diff = self.goal_location - envstate.last_state[0:3]
        last_delta_d = jnp.linalg.norm(last_diff)
        delta_distance = last_delta_d - delta_d
        distance_reward = self.reward_factors["delta_distance"] * delta_distance
        # self.last_reward_arr[14] = distance_reward
        reward += distance_reward

        thruster_penalty = -self.reward_factors["thruster_penalty"] * action[2] / 1000.
        # self.last_reward_arr[15] = thruster_penalty
        reward += thruster_penalty

        # reward_useful = {
        #     "goal": last_reward_arr_0,
        #     "goal_theta": last_reward_arr_1,
        #     "goal_psi": last_reward_arr_2,
        #     "velocity": last_reward_arr_13,
        #     "distance": last_reward_arr_14,
        #     "thruster": last_reward_arr_15,
        #
        #     "done_reached": last_reward_arr_8,
        #     "done_border": last_reward_arr_9,
        #     "done_attitude": last_reward_arr_10,
        #     "done_step": last_reward_arr_11,
        #     "done_collision": last_reward_arr_12,
        # }
        # return reward, self.last_reward_arr,reward_useful
        return reward

    # def generate_environment(self,rng: jax.random.PRNGKey):
    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """

        # curr_angle = (jnp.random.random(2) - 0.5) * 2 * jnp.array([jnp.pi / 2, jnp.pi])  # Water current direction
        # self.current = Current(mu=0.005, V_min=0.5, V_max=0.5, Vc_init=0.5,
        #                        alpha_init=curr_angle[0], beta_init=curr_angle[1], white_noise_std=0.0,
        #                        step_size=self.auv.step_size,current_on=self.current_on)
        # self.current = Current(remus100_env.karmen_current)
        # self.nu_c = self.current(self.auv.attitude, position=self.auv.position)

        self.goal_location = jnp.array(self.goal_point_)

        self.auv_init_position = jnp.array(self.start_point_)
        # Attitude
        # self.auv.attitude = self.generate_random_att(max_att_factor=0.7)
        self.auv_init_attitude = jnp.array([0, 0, 0.49 * jnp.pi])

        ###########
        CAPSULE_RADIUS = 1.0
        CAPSULE_HEIGHT = 4.0
        # Obstacles (only the capsule at goal location):
        cap = Capsule(position=jnp.array([0.0, 0.0, 0.0]),
                      radius=CAPSULE_RADIUS,
                      vec_top=jnp.array([0.0, 0.0, -CAPSULE_HEIGHT / 2.0]))
        self.capsules = [cap]
        self.obstacles = [*self.capsules]
        # Get vector pointing from goal location to capsule:
        vec = shape.vec_line_point(self.goal_location, cap.vec_top, cap.vec_bot)
        # Calculate heading at goal
        self.heading_goal_reached = geom.ssa(jnp.arctan2(vec[1], vec[0]))
        ###########

        ######
        CAPSULE_OBSTACLES_RADIUS = 2.0
        CAPSULE_OBSTACLES_HEIGHT = 2 * self.max_dist_from_goal
        CAPSULE_DISTANCE_FROM_CENTER = 6
        NUMBER_OF_CAPSULES = 1
        # Generate new capsules
        new_capsules = []
        # theta = jnp.random.rand() * 2 * jnp.pi
        # theta = jax.random.uniform(rng)* 2 * jnp.pi
        for i in range(NUMBER_OF_CAPSULES):
            # x = jnp.cos(theta) * CAPSULE_DISTANCE_FROM_CENTER
            # y = jnp.sin(theta) * CAPSULE_DISTANCE_FROM_CENTER
            x = -130.
            y = 0.
            # Create new capsule and append to list
            new_capsules.append(
                Capsule(position=jnp.array([x, y, 0.0]),
                        radius=CAPSULE_OBSTACLES_RADIUS,
                        vec_top=jnp.array([x, y, -CAPSULE_OBSTACLES_HEIGHT / 2.0]))
            )
        # Add Obstacles:
        self.capsules.extend(new_capsules)
        self.obstacles.extend(new_capsules)

        self.random_attitude = None
        self.random_position = None
        ######

    @partial(jax.jit, static_argnums=(0))
    def reset_env(
            self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))

        envstate = EnvState(
            state=jnp.concatenate([self.auv_init_position+0.01*jax.random.normal(key,(3,)), self.auv_init_attitude, jnp.array([0.0] * 6)]),
            state_dot=jnp.array([0.0] * 12),
            last_state=jnp.concatenate([self.auv_init_position, self.auv_init_attitude, jnp.array([0.0] * 6)]),
            u_actual=jnp.array([0.0] * 3),
            nu_c=jnp.array([0.0] * 6),
            time=0,
        )

        return self.get_obs(envstate), envstate

    @partial(jax.jit, static_argnums=(0))
    def get_obs(self, envstate: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        # return jnp.array([envstate.state, envstate.nu_c])
        diff = self.goal_location - envstate.state[0:3]
        delta_d = jnp.linalg.norm(diff)
        delta_theta = envstate.state[4] + (geom.ssa(jnp.arctan2(diff[2], jnp.linalg.norm(diff[:2]))))
        delta_psi = geom.ssa(jnp.arctan2(diff[1], diff[0]) - envstate.state[5])

        def custom_clip(x, min_val, max_val,name):
            """Custom clip function that warns when clipping occurs."""
            clipped = jnp.clip(x, min_val, max_val)

            def print_warning():


                jax.debug.print(f"error_my: Clipping occurred in '{name}'{x}")

                # def haha():
                #     jax.debug.print(f"error_my: !!!double error!!!Clipping occurred in '{name}'{x}")
                #     return clipped
                # jax.lax.cond(jnp.any(x>2.0), haha, lambda: clipped)
                # jax.lax.cond(jnp.any(x<-2.0), haha, lambda: clipped)
                return clipped

            # return jax.lax.cond(jnp.any(clipped != x), print_warning, lambda: clipped)
            return clipped

        # obs = jnp.zeros(self.n_observations, dtype=jnp.float32)
        # Distance from goal, contained within max_dist_from_goal before done
        obs_0 = custom_clip(1 - (jnp.log(0.5 * delta_d / self.max_dist_from_goal) / jnp.log(
            self.dist_goal_reached_tol / self.max_dist_from_goal)), 0, 1,"obs_0")
        # Pitch error delta_psi, will be between +90° and -90°
        obs_1 = custom_clip(delta_theta / (jnp.pi / 2), -1, 1,"obs_1")
        # Heading error delta_theta, will be between -180 and +180 degree, observation jump is not fixed here,
        # since it might be good to directly indicate which way to turn is faster to adjust heading
        obs_2 = custom_clip(delta_psi / jnp.pi, -1, 1,"obs_2")

        obs_3 = custom_clip(envstate.nu_c[0] / 2, -1, 1,"obs_3")  # Assuming in general current max. speed of 2m/s
        obs_4 = custom_clip(envstate.nu_c[1] / 2, -1, 1,"obs_4")
        obs_5 = custom_clip(envstate.nu_c[2] / 2, -1, 1,"obs_5")

        # ??? 竟然没有添加自身位置到obs里，amazing
        obs_6 = custom_clip(envstate.state[0] / self.bounding_box[0], -1, 1,"obs_6")
        obs_7 = custom_clip(envstate.state[1] / self.bounding_box[1], -1, 1,"obs_7")
        obs_8 = custom_clip(envstate.state[2] / self.bounding_box[2], -1, 1,"obs_8")
        obs_9 = custom_clip(envstate.state[3] / self.max_attitude, -1, 1, "obs_9")  # Roll
        # obs_6 = custom_clip(self.auv.attitude[0] / self.max_attitude, -1, 1)  # Roll
        obs_10 = custom_clip(envstate.state[4] / self.max_attitude, -1, 1, "obs_10")  # Pitch
        obs_11 = custom_clip(jnp.sin(envstate.state[5]), -1, 1, "obs_11")  # Yaw, expressed in two polar values to make
        obs_12 = custom_clip(jnp.cos(envstate.state[5]), -1, 1,
                            "obs_12")  # sure observation does not jump between -1 and 1
        # Deprecated, goal constraints removed - Erik, 30.06.2022
        # obs[3] = custom_clip(self.delta_heading_goal / jnp.pi, -1, 1)  # delta_psi_g, heading error for docking into goal
        obs_13 = custom_clip(envstate.state[6] / self.u_max, -1, 1,"obs_13")  # Surge Forward speed
        # obs_3 = custom_clip(self.auv.relative_velocity[0] / self.u_max, -1, 1)  # Surge Forward speed
        obs_14 = custom_clip(envstate.state[7] / self.v_max, -1, 1,"obs_14")  # Sway Side speed
        obs_15 = custom_clip(envstate.state[8] / self.w_max, -1, 1,"obs_15")  # Heave Vertical speed
        obs_16 = custom_clip(envstate.state[9] / self.p_max, -1, 1,"obs_16")  # Angular Velocities, roll rate
        obs_17 = custom_clip(envstate.state[10] / self.q_max, -1, 1,"obs_17")  # pitch rate
        obs_18 = custom_clip(envstate.state[11] / self.r_max, -1, 1,"obs_18")  # Yaw rate


        # def true_fn():
        #     return jnp.array([obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6, obs_7, obs_8,
        #                       obs_9, obs_10, obs_11, obs_12, obs_13, obs_14, obs_15, obs_16,
        #                       obs_17, obs_18, ])
        # def false_fn():
        #     obs_19 = jnp.clip(self.goal_location[0] / self.bounding_box[0], -1, 1)
        #     obs_20 = jnp.clip(self.goal_location[1] / self.bounding_box[1], -1, 1)
        #     obs_21 = jnp.clip(self.goal_location[2] / self.bounding_box[2], -1, 1)
        #     return jnp.array([obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6, obs_7, obs_8,
        #                       obs_9, obs_10, obs_11, obs_12, obs_13, obs_14, obs_15, obs_16,
        #                       obs_17, obs_18, obs_19, obs_20, obs_21])
        #
        # obs = jax.lax.cond(self.random_position is None, true_fn, false_fn)
        obs = jnp.array([obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6, obs_7, obs_8,
                              obs_9, obs_10, obs_11, obs_12, obs_13, obs_14, obs_15, obs_16,
                              obs_17, obs_18, ])
        return obs

    @partial(jax.jit, static_argnums=(0))
    def is_terminal(self, envstate: EnvState, params: EnvParams) -> jnp.ndarray:

        """
        Condition 0: Check if close to the goal
        Condition 1: Check if out of bounds for position
        Condition 2: Check if attitude (pitch, roll) too high
        Condition 3: Check if maximum time steps reached
        Condition 4: Check for collision

        :return: [if simulation is done, indexes of conditions that are true]
        """
        diff = self.goal_location - envstate.state[0:3]
        # self.delta_d 表示的是 AUV 与目标位置之间的距离
        delta_d = jnp.linalg.norm(diff)

        out_of_bounds = jnp.logical_or(
            jnp.logical_or(jnp.abs(envstate.state[0]) > self.bounding_box[0],
                           jnp.abs(envstate.state[1]) > self.bounding_box[1]),
            jnp.abs(envstate.state[2]) > self.bounding_box[2]
        )

        conditions = jnp.array([
            # Condition 0: Check if close to the goal
            delta_d < self.dist_goal_reached_tol,
            # Condition 1: Check if out of bounds for position
            out_of_bounds,
            # Condition 2: Check if attitude (pitch, roll) too high
            # jnp.any(jnp.abs(self.auv.attitude[:2]) > self.max_attitude),
            False,
            # Condition 3: Check if maximum time steps reached
            envstate.time >= self.max_timesteps,
            # Condition 4: Collision with obstacle (is updated earlier)
            # self.collision
            False
        ])

        # Check if any condition is true
        done = jnp.any(conditions)  # To satisfy environment checker

        def true_fn_0():
            # jax.debug.print("Goal reached, steps: {}", envstate.time)
            # print("Goal reached, steps: {}", envstate.time)
            pass

        def false_fn_0():
            # jax.debug.print("Goal donot reached, steps: {}", envstate.time)
            # print("Goal donot reached, steps: {}", envstate.time)
            pass


        def true_fn_2():
            # jax.debug.print("Attitude too high, steps:{} ", envstate.time)
            pass
        # jax.lax.cond(conditions[0], true_fn_0, lambda: None)
        # jax.lax.cond(jnp.any(conditions[0]), true_fn_0, false_fn_0)
        # jax.lax.cond(conditions[2], true_fn_2, lambda: None)
        #     # If goal reached
        #     if conditions[0]:
        #         goal_reached = True
        #     if conditions[2]:
        #         jax.debug.print("Attitude too high, steps: ", self.t_steps)
        #     indices = jnp.where(conditions)[0]
        #     # wandb.log({ "logs/done_condition(0:goal,1:border,3:step,4：collision)": indices[0]})
        #     # Return also the indexes of which cond is activated
        # cond_idx = [i for i, x in enumerate(conditions) if x]
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "remus_v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    # def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
    #     """Action space of the environment."""
    #     return spaces.Discrete(2)
    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""

        return spaces.Box(
            low=jnp.array([-1.,-1.,-1.]),
            high=jnp.array([1.,1.,1.]),
            shape=(3,), dtype=jnp.float32
        )

        # self.action_space = Box(low=jnp.concatenate(
        #     (self.u_bound[:, 0][0:-1], jnp.array([self.config["thruster_min"]]))),
        #     high=jnp.concatenate(
        #         (self.u_bound[:, 1][0:-1], jnp.array([env_config["thruster"]]))),
        #     dtype=jnp.float32)
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        obs_low = - jnp.ones(self.n_observations)
        obs_low = obs_low.at[0].set(0.0)
        obs_high = jnp.ones(self.n_observations)
        return spaces.Box(obs_low, obs_high, (self.n_observations,), dtype=jnp.float32)

        # # # Except for delta distance and rays, observation is between -1 and 1
        # obs_low = - jnp.ones(self.n_observations)
        # obs_low = obs_low.at[0].set(0.0)
        # # obs_low = obs_low.at[self.n_obs_without_radar:].set[0]
        # self.observation_space = Box(low=obs_low,
        #                              high=jnp.ones(self.n_observations),
        #                              dtype=jnp.float32)
        # self.observation = jnp.zeros(self.n_observations)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                [1000.0]*12,
                [1000.0]*12,
                [1000.0]*12,
                [1000.0]*3,
                [1000.0] * 6,
                1000
            ]
        )
        return spaces.Dict(
            {
                "state": spaces.Box(-high[0], high[0], (), jnp.float32),
                "state_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "last_state": spaces.Box(-high[2], high[2], (), jnp.float32),
                "u_actual": spaces.Box(-high[3], high[3], (), jnp.float32),
                "nu_c": spaces.Box(-high[4], high[4], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )




if __name__ == '__main__':  # todo bug solve
    # karmen = remus100_env.karmen_current
    # haha = karmen.generate_current(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    # haha = karmen.generate_current(jnp.array(-27.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    print("-------------------")
    # current = Current(remus100_env.karmen_current)
    #
    # t = time.time()
    # haha = current(jnp.array([0.1, 0.2, 0.3]),jnp.array([0.1, 0.2, 0.3]))
    # print(time.time() - t)
    # print(haha)
    #
    # t = time.time()
    # haha = current(jnp.array([0.1, 0.2, 0.3]),jnp.array([0.1, 0.2, 0.3]))
    # print(time.time() - t)
    # print(haha)

    print("-------------------")

    from gym_dockauv.config.env_config import PREDICT_CONFIG, MANUAL_CONFIG, TRAIN_CONFIG, REGISTRATION_DICT, \
        TRAIN_CONFIG_remus_Karman

    used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
    used_TRAIN_CONFIG["vehicle"] = "remus100"
    start_point = [-0, -10, 0]
    goal_point = [-0, 10, 0]
    used_TRAIN_CONFIG["start_point"] = start_point
    used_TRAIN_CONFIG["goal_point"] = goal_point
    used_TRAIN_CONFIG["bounding_box"] = [26, 18, 20]
    # 计算二范数
    used_TRAIN_CONFIG["max_dist_from_goal"] = jnp.linalg.norm(jnp.array(goal_point) - jnp.array(start_point))
    used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.05 * jnp.linalg.norm(jnp.array(goal_point) - jnp.array(start_point))
    used_TRAIN_CONFIG["max_timesteps"] = 999

    used_TRAIN_CONFIG["title"] = "Training Run"


    @dataclass
    class args:
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        current_on: bool = True
        tau: float = 0.5

        w_velocity: float = 0.1
        thruster_penalty: float = 1.0
        thruster: float = 400
        thruster_min: float = 0.


    used_TRAIN_CONFIG["current_on"] = args.current_on
    used_TRAIN_CONFIG["reward_factors"]["w_velocity"] = args.w_velocity
    used_TRAIN_CONFIG["reward_factors"]["thruster_penalty"] = args.thruster_penalty
    used_TRAIN_CONFIG["thruster"] = args.thruster
    used_TRAIN_CONFIG["thruster_min"] = args.thruster_min

    environment = remus_v1(used_TRAIN_CONFIG)
    key_true = jax.random.PRNGKey(0)
    initial_state = EnvState(
        state=jnp.zeros(12),  # Assuming a state vector of length 12
        state_dot=jnp.zeros(12),
        last_state=jnp.zeros(12),
        u_actual=jnp.zeros(3),  # Assuming control vector of length 4
        nu_c=jnp.zeros(3),  # Assuming nu_c vector of length 3
        time=0
    )
    action = jnp.array([1.0, 0.0, 1.0])  # Dummy action
    params = EnvParams()

    key_true ,key = jax.random.split(key_true, 2)
    obs, new_state, reward, done, info = environment.step_env(key, initial_state, action, params)
    print("Observation:", obs)
    print("New State:", new_state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    key_true ,key = jax.random.split(key_true, 2)
    obs, state = environment.reset(key, params)
    print("reset:")
    print("Observation:", obs)
    print("State:", state)

    key_true ,key = jax.random.split(key_true, 2)
    obs, new_state, reward, done, info = environment.step(key, initial_state, action, params)
    print("step")
    print("Observation:", obs)
    print("reward",reward)
    print("Done:", done)
    print("Info:", info)

    key_true ,key = jax.random.split(key_true, 2)
    action = environment.action_space(params).sample(key)
    print("Action:", action)

    print("******************")

    # key ,key_next = jax.random.split(key, 2)
    # environment.generate_environment(key_next)

    print("-------------------")

    # reward = Reward()
    #
    # t = time.time()
    # haha = reward.log_precision(1.0,3.0,2.0)
    # print(time.time() - t)
    # print(haha)
    #
    # # haha = reward.obstacle_avoidance(jnp.array(1.0),jnp.array(1.0),jnp.array(1.0),1.0,1.0,1.0,1.0,1.0)
    # t = time.time()
    # haha = reward.obstacle_avoidance(jnp.array([1.0,1.0]),jnp.array([1.0,1.0]),jnp.array([1.0,1.0]),1.0,1.0,1.0,1.0,1.0)
    # print(time.time() - t)
    # print(haha)
    #
    # t = time.time()
    # haha = reward.disc_goal_constraints(1.0,1.0)
    # print(time.time() - t)
    # print(haha)

    print("-------------------")
