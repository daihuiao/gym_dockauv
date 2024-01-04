#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remus100.py:  

   Class for the Remus 100 cylinder-shaped autonomous underwater vehicle (AUV), 
   which is controlled using a tail rudder, stern planes and a propeller. The 
   length of the AUV is 1.6 m, the cylinder diameter is 19 cm and the 
   mass of the vehicle is 31.9 kg. The maximum speed of 2.5 m/s is obtained 
   when the propeller runs at 1525 rpm in zero currents.
       
   remus100()                           
       Step input, stern plane, rudder and propeller revolution     
   
    remus100('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c)
        z_d:    desired depth (m), positive downwards
        psi_d:  desired yaw angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c:    current speed (m/s)
        beta_c: current direction (deg)                  

Methods:
        
    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns 
        nu[k+1] and u_actual[k+1] using Euler's method. The control input is:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]

    u = depthHeadingAutopilot(eta,nu,sampleTime) 
        Simultaneously control of depth and heading using two controllers of 
        PID type. Propeller rpm is given as a step command.
       
    u = stepInput(t) generates tail rudder, stern planes and RPM step inputs.   
       
References: 
    
    B. Allen, W. S. Vorus and T. Prestero, "Propulsion system performance 
         enhancements on REMUS AUVs," OCEANS 2000 MTS/IEEE Conference and 
         Exhibition. Conference Proceedings, 2000, pp. 1869-1873 vol.3, 
         doi: 10.1109/OCEANS.2000.882209.    
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
         Control. 2nd. Edition, Wiley. URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import copy

import numpy as np
import math
import sys
from python_vehicle_simulator.lib.control import PIDpolePlacement
from python_vehicle_simulator.lib.gnc import crossFlowDrag, forceLiftDrag, Hmtrx, m2c, gvect, ssa, attitudeEuler
from gym_dockauv.utils import geomutils as geom


# Class Vehicle
class remus100:
    """
    remus100()
        Rudder angle, stern plane and propeller revolution step inputs
        
    remus100('depthHeadingAutopilot',z_d,psi_d,n_d,V_c,beta_c) 
        Depth and heading autopilots
        
    Inputs:
        z_d:    desired depth, positive downwards (m)
        psi_d:  desired heading angle (deg)
        n_d:    desired propeller revolution (rpm)
        V_c:    current speed (m/s)
        beta_c: current direction (deg)
    """

    def __init__(
            self,
            controlSystem="stepInput",
            r_z=0,
            r_psi=0,
            r_rpm=0,
            V_current=0,
            beta_current=0,
    ):
        self.DOF = 6  # degrees of freedom

        # Constants
        self.D2R = math.pi / 180  # deg2rad
        self.rho = 1026  # density of water (kg/m^3)
        g = 9.81  # acceleration of gravity (m/s^2)

        if controlSystem == "depthHeadingAutopilot":
            self.controlDescription = (
                    "Depth and heading autopilots, z_d = "
                    + str(r_z)
                    + ", psi_d = "
                    + str(r_psi)
                    + " deg"
            )

        else:
            self.controlDescription = (
                "Step inputs for stern planes, rudder and propeller")
            controlSystem = "stepInput"

        self.ref_z = r_z
        self.ref_psi = r_psi
        self.ref_n = r_rpm
        self.V_c = V_current
        self.beta_c = beta_current * self.D2R
        self.controlMode = controlSystem

        # Initialize the AUV model 
        self.name = (
            "Remus 100 cylinder-shaped AUV (see 'remus100.py' for more details)")
        self.L = 1.6  # length (m)
        self.diam = 0.19  # cylinder diameter (m)

        self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # velocity vector
        self.u_actual = np.array([0, 0, 0], float)  # control input vector

        self.controls = [
            "Tail rudder (deg)",
            "Stern plane (deg)",
            "Propeller revolution (rpm)"
        ]
        self.dimU = len(self.controls)

        # Actuator dynamics
        self.deltaMax_r = 30 * self.D2R  # max rudder angle (rad)
        self.deltaMax_s = 30 * self.D2R  # max stern plane angle (rad)
        self.nMax = 1525  # max propeller revolution (rpm)
        self.T_delta = 1.0  # rudder/stern plane time constant (s)
        self.T_n = 1.0  # propeller time constant (s)

        if r_rpm < 0.0 or r_rpm > self.nMax:
            sys.exit("The RPM value should be in the interval 0-%s", (self.nMax))

        if r_z > 100.0 or r_z < 0.0:
            sys.exit('desired depth must be between 0-100 m')

            # Hydrodynamics (Fossen 2021, Section 8.4.2)
        # S is a reference area usually chosen as the area of the vehicle as if it were projected down onto the ground below it.
        self.S = 0.7 * self.L * self.diam  # S = 70% of rectangle L * diam
        a = self.L / 2  # semi-axes
        b = self.diam / 2
        self.r_bg = np.array([0, 0, 0.02], float)  # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)  # CB w.r.t. to the CO

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)   
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42  # from Allen et al. (2000)
        self.CD_0 = Cd * math.pi * b ** 2 / self.S

        # Rigid-body mass matrix expressed in CO
        m = 4 / 3 * math.pi * self.rho * a * b ** 2  # mass of spheriod
        Ix = (2 / 5) * m * b ** 2  # moment of inertia
        Iy = (1 / 5) * m * (a ** 2 + b ** 2)
        Iz = Iy
        MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])  # MRB expressed in the CG
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg  # MRB expressed in the CO # 61页上

        # Weight and buoyancy
        self.W = m * g
        self.B = self.W

        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3
        MA_44 = r44 * Ix  # ？

        # Lamb's k-factors
        e = math.sqrt(1 - (b / a) ** 2)
        alpha_0 = (2 * (1 - e ** 2) / pow(e, 3)) * (0.5 * math.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / (e ** 2) - (1 - e ** 2) / (2 * pow(e, 3)) * math.log((1 + e) / (1 - e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = pow(e, 4) * (beta_0 - alpha_0) / (
                (2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * (beta_0 - alpha_0)))

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy])  # 8.84

        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                self.M[3][3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                 self.M[4][4])

        # Tail rudder parameters (single)
        # self.CL_delta_r = 0.5  # rudder lift coefficient
        #todo dai: 转弯太慢了，改了，很假，望州之
        self.CL_delta_r = 5  # rudder lift coefficient
        self.A_r = 2 * 0.10 * 0.05  # rudder area (m2)
        self.x_r = -a  # rudder x-position (m)

        # Stern-plane paramaters (double)
        # self.CL_delta_s = 0.7  # stern-plane lift coefficient
        #todo dai: 转弯太慢了，改了，很假，
        self.CL_delta_s = 7  # stern-plane lift coefficient
        self.A_s = 2 * 0.10 * 0.05  # stern-plane area (m2)
        self.x_s = -a  # stern-plane z-position (m)

        # Low-speed linear damping matrix parameters
        self.T_surge = 20  # time constant in surge (s)
        self.T_sway = 20  # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3  # relative damping ratio in roll
        self.zeta_pitch = 0.8  # relative damping ratio in pitch
        self.T_yaw = 5  # time constant in yaw (s)

        # Heading autopilot
        self.wn_psi = 0.5  # PID pole placement parameters
        self.zeta_psi = 1
        self.r_max = 1 * math.pi / 180  # maximum yaw rate 
        self.psi_d = 0  # position, velocity and acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = self.wn_psi / 5  # desired natural frequency
        self.zeta_d = 1  # desired realtive damping ratio

        self.e_psi_int = 0  # yaw angle error integral state

        # Depth autopilot
        self.wn_d_z = 1 / 20  # desired natural frequency, reference model
        self.Kp_z = 0.1  # heave proportional gain, outer loop
        self.T_z = 100.0  # heave integral gain, outer loop
        self.Kp_theta = 1.0  # pitch PID controller
        self.Kd_theta = 3.0
        self.Ki_theta = 0.1

        self.z_int = 0  # heave position integral state
        self.z_d = 0  # desired position, LP filter initial state
        self.theta_int = 0  # pitch angle integral state

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime,nu_c):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the AUV equations of motion using Euler's method.
        舵（Rudder）主要用于控制船舶的航向，即左右转向。它通常位于船舶的尾部，可以左右摆动，通过改变水流方向来改变船舶的航向。
        船尾平面（Stern-plane），有时也称为潜艇的水平尾翼，用于控制船舶的俯仰运动，即上升和下降。
        它们位于船舶尾部的上下位置，并可以上下摆动，通过改变水流的垂直方向来控制船舶的深度。
        """
#todo dai 默认这里的nu_c是0,如果想要加入水流的影响，需要在这里加入水流的影响，把下面的洋流替换，nu_c=[0,0,0,0,0,0]
        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge velocity
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity
        Dnu_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0, 0, 0, 0], float)  # derivative
        nu_r = nu - nu_c  # relative velocity
        alpha = math.atan2(nu_r[2], nu_r[0])  # angle of attack
        U = math.sqrt(nu[0] ** 2 + nu[1] ** 2 + nu[2] ** 2)  # vehicle speed
        U_r = math.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2 + nu_r[2] ** 2)  # relative speed

        """     
        1.`u_c`和`v_c`分别计算了水流的纵向（surge）和横向（sway）速度分量，基于水流速度`self.V_c`和相对船舶航向的水流方向
        `self.beta_c`与船舶航向`eta[5]`的差值。
        2.`nu_c`是水流速度向量，只考虑水平面上的运动。
        3.`Dnu_c`是水流速度变化率的向量，考虑了船舶偏航角速度`nu[5]`对水流速度的影响。第一个分量 nu[5] * v_c 计算了由于船舶偏航角速度 nu[5]
        和横向水流速度 v_c 相互作用产生的纵向水流速度变化。 第二个分量 -nu[5] * u_c 计算了由于偏航角速度和纵向水流速度 u_c 相互作用产生的横向水流速度变化。
        4.`nu_r`是船舶相对于水流的速度向量。
        5.`alpha`是攻角，即船舶前进方向与相对水流方向之间的角度。
        6.`U`是船舶的绝对速度大小。
        7.`U_r`是船舶相对于水流的速度大小。
        """

        # Commands and actual control signals
        delta_r_c = u_control[0]  # commanded tail rudder (rad)
        delta_s_c = u_control[1]  # commanded stern plane (rad)
        n_c = u_control[2]  # commanded propeller revolution (rpm)

        delta_r = u_actual[0]  # actual tail rudder (rad)
        delta_s = u_actual[1]  # actual stern plane (rad)
        n = u_actual[2]  # actual propeller revolution (rpm)

        # Amplitude saturation of the control signals
        if abs(delta_r) >= self.deltaMax_r:
            delta_r = np.sign(delta_r) * self.deltaMax_r

        if abs(delta_s) >= self.deltaMax_s:
            delta_s = np.sign(delta_s) * self.deltaMax_s

        if abs(n) >= self.nMax:
            n = np.sign(n) * self.nMax

            # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        D_prop = 0.14  # propeller diameter corresponding to 5.5 inches
        t_prop = 0.1  # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps) 
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632

        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        KT_max = 0.1798
        KQ_max = 0.0312

        """        这段代码计算了螺旋桨的性能系数，KT（推力系数）和KQ（扭矩系数），这些系数是通过推进数（advance
        number）Ja计算得到的。

        Ja计算了推进数，它是螺旋桨的有效进速(Va)除以螺旋桨转速(n)和螺旋桨直径(D_prop)的乘积。
        Va是螺旋桨的有效进速，计算为船舶速度(U)的一个固定比例（此处为0.944，根据Allen等人2000年的研究）。
        D_prop是螺旋桨直径，t_prop是推力减额数。n_rps是螺旋桨每秒转数，由每分钟转数(n)转换得来。
        KT_0和KQ_0是在无推进数时的螺旋桨推力系数和扭矩系数的估计值。KT_max和KQ_max是在最大推进数(Ja_max)
        下的螺旋桨推力系数和扭矩系数的估计值。"""

        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  

        if n_rps > 0:  # forward thrust

            X_prop = self.rho * pow(D_prop, 4) * (
                    KT_0 * abs(n_rps) * n_rps + (KT_max - KT_0) / Ja_max *
                    (Va / D_prop) * abs(n_rps))
            K_prop = self.rho * pow(D_prop, 5) * (
                    KQ_0 * abs(n_rps) * n_rps + (KQ_max - KQ_0) / Ja_max *
                    (Va / D_prop) * abs(n_rps))

        else:  # reverse thrust (braking)

            X_prop = self.rho * pow(D_prop, 4) * KT_0 * abs(n_rps) * n_rps
            K_prop = self.rho * pow(D_prop, 5) * KQ_0 * abs(n_rps) * n_rps
        """        当螺旋桨正转（向前推进）时，代码计算螺旋桨产生的推力（X_prop）和由螺旋桨引起的滚转力矩（K_prop）：
        X_prop和K_prop的计算基于线性近似的推力系数（KT）和扭矩系数（KQ）。这些系数随着推进数（Ja）的变化而变化。
        KT和KQ通过插值计算得出，基于零推进数（KT_0，KQ_0）和最大推进数（KT_max，KQ_max）下的系数值。
        推力（X_prop）和扭矩（K_prop）分别是与螺旋桨直径的四次幂和五次幂成比例，考虑了水的密度、螺旋桨的转速和有效进速。"""

        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = m2c(self.MRB, nu_r)
        CA = m2c(self.MA, nu_r)

        # Nonlinear quadratic velocity terms in pitch and yaw (Munk moments) 
        # are set to zero since only linear damping is used
        CA[4][0] = 0
        CA[4][3] = 0
        CA[5][0] = 0
        CA[5][1] = 0

        C = CRB + CA

        """        这段代码涉及到船舶动力学中的科里奥利力和离心力矩阵的计算，这些力矩阵与船舶的刚体质量（MRB）和附加质量（MA）相关：

        CRB = m2c(self.MRB, nu_r)：计算刚体质量矩阵（MRB）相关的科里奥利 / 离心力矩阵。m2c
        函数基于船舶的运动状态nu_r和MRB矩阵来计算。

        CA = m2c(self.MA, nu_r)：计算附加质量矩阵（MA）相关的科里奥利 / 离心力矩阵。这同样使用m2c函数，但是基于附加质量矩阵。

        对于俯仰和偏航运动的非线性二次速度项（也称为蒙克力矩）设置为零，因为这里只使用线性阻尼。这是通过将CA矩阵的特定元素设置为0来实现的。

        C = CRB + CA：将刚体质量和附加质量相关的科里奥利 / 离心力矩阵相加，得到总的科里奥利 / 离心力矩阵C。"""

        # Dissipative forces and moments
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
        ])

        D[0][0] = D[0][0] * math.exp(-3 * U_r)  # For DOF 1,2,6 the D elements
        D[1][1] = D[1][1] * math.exp(-3 * U_r)  # go to zero at higher speeds, i.e.
        D[5][5] = D[5][5] * math.exp(-3 * U_r)  # drag and lift/drag dominate

        """        这段代码计算了耗散力和力矩矩阵D，它是一个对角矩阵，代表船舶在不同方向的阻尼效应：
        
        对于船舶的纵向（surge）、横向（sway）和垂直（heave）运动，阻尼计算为质量矩阵的对应元素除以时间常数
        T_surge、T_sway和T_heave。对于滚转（roll）和俯仰（pitch）运动，阻尼考虑了相对阻尼比zeta_roll和zeta_pitch，以及相应的自然频率
        w_roll和w_pitch。对于偏航（yaw）运动，阻尼同样是质量矩阵的对应元素除以时间常数T_yaw。
        在一定速度下，纵向、横向和偏航的阻尼因子会因速度的增加而减小，这通过乘以exp(-3 * U_r)实现，其中U_r是相对速度。"""

        tau_liftdrag = forceLiftDrag(self.diam, self.S, self.CD_0, alpha, U_r)  # 阻力 升力
        tau_crossflow = crossFlowDrag(self.L, self.diam, self.diam, nu_r)  # 侧向水流（横流）作用下的阻力

        # Restoring forces and moments
        g = gvect(self.W, self.B, eta[4], eta[3], self.r_bg, self.r_bb)  # 计算一个潜在水中物体在任意点CO（通常是质心或者流体动力中心）的恢复力矢量。

        # Horizontal- and vertical-plane relative speed
        U_rh = math.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2)
        U_rv = math.sqrt(nu_r[0] ** 2 + nu_r[2] ** 2)

        # Rudder and stern-plane drag
        X_r = -0.5 * self.rho * U_rh ** 2 * self.A_r * self.CL_delta_r * delta_r ** 2
        X_s = -0.5 * self.rho * U_rv ** 2 * self.A_s * self.CL_delta_s * delta_s ** 2
        """        这段代码计算了与舵（Rudder）和船尾平面（Stern-plane）相关的阻力。`U_rh` 是水平平面上的相对速度，
        结合了船舶前进速度和侧滑速度。`U_rv` 是垂直平面上的相对速度，结合了船舶前进速度和垂直速度。
        `X_r` 是因舵角而产生的阻力，而 `X_s` 是因船尾平面角而产生的阻力。这些阻力由流体动力学中的二次阻力公式计算得出，
        该公式涉及流体密度（`self.rho`）、相对速度平方、参考面积（`self.A_r` 对于舵，`self.A_s` 对于船尾平面）
        和阻力系数（`self.CL_delta_r` 对于舵，`self.CL_delta_s` 对于船尾平面）。这些阻力在船舶运动方程中起着重要作用，
        因为它们直接影响船舶的运动和所需的推进力。"""

        # Rudder sway force 
        Y_r = -0.5 * self.rho * U_rh ** 2 * self.A_r * self.CL_delta_r * delta_r

        # Stern-plane heave force
        Z_s = -0.5 * self.rho * U_rv ** 2 * self.A_s * self.CL_delta_s * delta_s
        """        Y_r 表示舵产生的侧向力（sway force），这个力使得船舶能够向左或向右移动（偏航）。Z_s 表示船尾平面产生的垂直力（heave force），
        这个力控制着船舶的上升或下沉（俯仰）。这两个力分别是通过改变水流方向，由舵和船尾平面在其相应的控制角度（delta_r 和 delta_s）下产生的。
        这些力对于航行器的稳定性和操控性至关重要，它们帮助航行器在水中保持预定的路径和深度。"""

        # Generalized force vector
        tau = np.array([
            (1 - t_prop) * X_prop + X_r + X_s,
            Y_r,
            Z_s,
            K_prop / 10,  # scaled down by a factor of 10 to match exp. results
            self.x_s * Z_s,
            self.x_r * Y_r
        ], float)
        """        这段代码中的 tau 是一个广义力向量，包含了作用于船舶的力和力矩：
        第一项：(1-t_prop) * X_prop + X_r + X_s 是沿船舶前进方向的总力，考虑了螺旋桨推进力 X_prop，扣除了推进力减额系数 t_prop 的影响，
        并加上舵产生的阻力 X_r 和船尾平面产生的阻力 X_s。
        第二项：Y_r 是舵产生的侧向力，用于船舶的偏航控制。
        第三项：Z_s 是船尾平面产生的垂直力，用于控制船舶的深度。
        第四项：K_prop / 10 是螺旋桨产生的力矩，这里除以10是为了匹配实验结果的规模。
        第五项：self.x_s * Z_s 是船尾平面产生的俯仰力矩，self.x_s 是船尾平面的位置。
        第六项：self.x_r * Y_r 是舵产生的偏航力矩，self.x_r 是舵的位置。
        这些力和力矩共同决定了船舶在水中的运动状态，包括平动和转动。"""

        # AUV dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C + D, nu_r) - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)  # 牛顿第二定律，求a
        """        tau_sum 是所有外力和力矩的总和。它包括前面计算的广义力 tau，升力和阻力 tau_liftdrag，横流阻力 tau_crossflow，以及由于船舶运动产生的科里奥利力
        和阻尼力（通过 C+D 乘以相对速度 nu_r 得到），最后减去重力和浮力产生的作用力 g。
        nu_dot 是船舶运动状态（速度和角速度）的变化率。Dnu_c 表示由于当前流体速度产生的影响，self.Minv 是船舶质量矩阵的逆，
        用于计算由于外部作用力 tau_sum 引起的加速度。"""

        # Actuator dynamics
        delta_r_dot = (delta_r_c - delta_r) / self.T_delta
        delta_s_dot = (delta_s_c - delta_s) / self.T_delta
        n_dot = (n_c - n) / self.T_n

        """        这段代码描述了船舶操纵装置（舵、船尾平面和螺旋桨）的动态特性。delta_r_dot、delta_s_dot
        和n_dot分别是舵角、船尾平面角和螺旋桨转速的变化率。这些变化率计算基于期望值（delta_r_c、delta_s_c
        和n_c）与当前实际值（delta_r、delta_s和n）之间的差异，并除以各自的时间常数（self.T_delta对于舵和船尾平面，self.T_n
        对于螺旋桨）。这反映了操纵装置响应控制命令的动态过程。"""

        # Forward Euler integration [k+1]
        nu += sampleTime * nu_dot
        delta_r += sampleTime * delta_r_dot
        delta_s += sampleTime * delta_s_dot
        n += sampleTime * n_dot

        u_actual = np.array([delta_r, delta_s, n], float)

        return nu, u_actual, nu_dot

    def reset(self):
        self.t = 0  # initial simulation time

        # # Initial state vectors
        # self.eta = np.array([0, 0, 0, 0, 0, 0], float)  # position/attitude, user editable
        # self.nu = np.array([0, 0, 0, 0, 0, 0], float)  # velocity, defined by vehicle class
        # self.u_actual = np.array([0, 0, 0], float)  # actual inputs, defined by vehicle class

        # Initialization of table used to store the simulation data
        self.simData = np.empty([0, 2 * self.DOF + 2 * self.dimU], float)

    def remus_solver(self, u_control, eta, nu, nu_c, u_actual, N=5, sampleTime=0.02):
        # eta_old = copy.deepcopy(eta)
        for i in range(0, N ):
            # Store simulation data in simData
            # signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
            # self.simData = np.vstack([self.simData, signals])

            # 这里返回的都是AUV坐标系计算的值
            [nu, u_actual, nu_dot] = self.dynamics(eta, nu, u_actual, u_control, sampleTime,nu_c)

            # 欧拉方法计算位置和速度,函数里使用2.53进行坐标系转化
            eta = attitudeEuler(eta, nu, sampleTime)

        self.t += N * sampleTime  # simulation time

        # Store simulation time vector
        # simTime = np.arange(start=0, stop=self.t + sampleTime, step=sampleTime)[:, None]
        # return (simTime, self.simData)

        # nu = nu_r + nu_c

        state_dot = np.zeros(12)
        # Kinematic Model
        state_dot[:6] = geom.J(eta).dot(nu)
        state_dot[6:] = nu_dot
        return np.concatenate((eta, nu), axis=0), u_actual, state_dot

    def stepInput(self, t):
        """
        u_c = stepInput(t) generates step inputs.
                     
        Returns:
            
            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]
        """
        delta_r = 5 * self.D2R  # rudder angle (rad)
        delta_s = -5 * self.D2R  # stern angle (rad)
        n = 1525  # propeller revolution (rpm)

        if t > 100:
            delta_r = 0

        if t > 50:
            delta_s = 0

        u_control = np.array([delta_r, delta_s, n], float)

        return u_control

    def depthHeadingAutopilot(self, eta, nu, sampleTime):
        """
        [delta_r, delta_s, n] = depthHeadingAutopilot(eta,nu,sampleTime) 
        simultaneously control the heading and depth of the AUV using control
        laws of PID type. Propeller rpm is given as a step command.
        
        Returns:
            
            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         n          propeller revolution (rpm) ]
            
        """
        z = eta[2]  # heave position (depth)
        theta = eta[4]  # pitch angle
        psi = eta[5]  # yaw angle
        q = nu[4]  # pitch rate
        r = nu[5]  # yaw rate
        e_psi = psi - self.psi_d  # yaw angle tracking error
        e_r = r - self.r_d  # yaw rate tracking error
        z_ref = self.ref_z  # heave position (depth) setpoint
        psi_ref = self.ref_psi * self.D2R  # yaw angle setpoint

        #######################################################################
        # Propeller command
        #######################################################################
        n = self.ref_n

        #######################################################################            
        # Depth autopilot (succesive loop closure)
        #######################################################################
        # LP filtered desired depth command
        self.z_d = math.exp(-sampleTime * self.wn_d_z) * self.z_d \
                   + (1 - math.exp(-sampleTime * self.wn_d_z)) * z_ref

        # PI controller    
        theta_d = self.Kp_z * ((z - self.z_d) + (1 / self.T_z) * self.z_int)
        delta_s = -self.Kp_theta * ssa(theta - theta_d) - self.Kd_theta * q \
                  - self.Ki_theta * self.theta_int

        # Euler's integration method (k+1)
        self.z_int += sampleTime * (z - self.z_d);
        self.theta_int += sampleTime * ssa(theta - theta_d);

        #######################################################################
        # Heading autopilot (PID controller)
        #######################################################################

        wn = self.wn_psi  # PID natural frequency
        zeta = self.zeta_psi  # PID natural relative damping factor
        wn_d = self.wn_d  # reference model natural frequency
        zeta_d = self.zeta_d  # reference model relative damping factor

        m = self.M[5][5]
        d = 0
        k = 0

        # PID feedback controller with 3rd-order reference model
        [delta_r, self.e_psi_int, self.psi_d, self.r_d, self.a_d] = \
            PIDpolePlacement(
                self.e_psi_int,
                e_psi, e_r,
                self.psi_d,
                self.r_d,
                self.a_d,
                m,
                d,
                k,
                wn_d,
                zeta_d,
                wn,
                zeta,
                psi_ref,
                self.r_max,
                sampleTime
            )

        # Euler's integration method (k+1)
        self.e_psi_int += sampleTime * ssa(psi - self.psi_d);

        u_control = np.array([delta_r, delta_s, n], float)

        return u_control
