#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from functools import partial
import jax
# import numpy as jnp
from jax import numpy as jnp
import jax.lax as lax

# import math
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
        self.D2R = jnp.pi / 180  # deg2rad
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

        self.nu = jnp.array([0, 0, 0, 0, 0, 0], float)  # velocity vector
        self.u_actual = jnp.array([0, 0, 0], float)  # control input vector

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
        self.r_bg = jnp.array([0, 0, 0.02], float)  # CG w.r.t. to the CO
        self.r_bb = jnp.array([0, 0, 0], float)  # CB w.r.t. to the CO

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)   
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42  # from Allen et al. (2000)
        self.CD_0 = Cd * jnp.pi * b ** 2 / self.S

        # Rigid-body mass matrix expressed in CO
        m = 4 / 3 * jnp.pi * self.rho * a * b ** 2  # mass of spheriod
        Ix = (2 / 5) * m * b ** 2  # moment of inertia
        Iy = (1 / 5) * m * (a ** 2 + b ** 2)
        Iz = Iy
        MRB_CG = jnp.diag(jnp.array([m, m, m, Ix, Iy, Iz]))  # MRB expressed in the CG
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg  # MRB expressed in the CO # 61页上

        # Weight and buoyancy
        self.W = m * g
        self.B = self.W

        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3
        MA_44 = r44 * Ix  # ？

        # Lamb's k-factors
        e = jnp.sqrt(1 - (b / a) ** 2)
        alpha_0 = (2 * (1 - e ** 2) / pow(e, 3)) * (0.5 * jnp.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / (e ** 2) - (1 - e ** 2) / (2 * pow(e, 3)) * jnp.log((1 + e) / (1 - e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = pow(e, 4) * (beta_0 - alpha_0) / (
                (2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * (beta_0 - alpha_0)))

        # Added mass system matrix expressed in the CO
        self.MA = jnp.diag(jnp.array([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy]))  # 8.84

        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = jnp.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = jnp.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                self.M[3][3])
        self.w_pitch = jnp.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
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
        self.r_max = 1 * jnp.pi / 180  # maximum yaw rate 
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

    @partial(jax.jit, static_argnums=(0))
    def dynamics(self, eta, nu, u_actual, u_control, sampleTime,nu_c):

        u_c, v_c, should_be_zero, _, _, _ = nu_c

        nu_c = jnp.array([u_c, v_c, should_be_zero, 0, 0, 0], float)  # current velocity
        Dnu_c = jnp.array([nu[5] * v_c, -nu[5] * u_c, 0, 0, 0, 0], float)  # derivative

        nu_r = nu - nu_c  # relative velocity
        alpha = jnp.arctan2(nu_r[2], nu_r[0])  # angle of attack
        U = jnp.sqrt(nu[0] ** 2 + nu[1] ** 2 + nu[2] ** 2)  # vehicle speed
        U_r = jnp.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2 + nu_r[2] ** 2)  # relative speed



        # Commands and actual control signals
        delta_r_c = u_control[0]  # commanded tail rudder (rad)
        delta_s_c = u_control[1]  # commanded stern plane (rad)
        n_c = u_control[2]  # commanded propeller revolution (rpm)

        delta_r = u_actual[0]  # actual tail rudder (rad)
        delta_s = u_actual[1]  # actual stern plane (rad)
        n = u_actual[2]  # actual propeller revolution (rpm)


        delta_r = jnp.clip(delta_r, - self.deltaMax_r, self.deltaMax_r)


        delta_s = jnp.clip(delta_s, - self.deltaMax_s, self.deltaMax_s)


        n = jnp.clip(n, - self.nMax, self.nMax)

        D_prop = 0.14  # propeller diameter corresponding to 5.5 inches
        t_prop = 0.1  # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps)
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632


        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3)
        KT_max = 0.1798
        KQ_max = 0.0312


        def forward_thrust(n_rps):
            X_prop = self.rho * pow(D_prop, 4) * (
                    KT_0 * abs(n_rps) * n_rps + (KT_max - KT_0) / Ja_max *
                    (Va / D_prop) * abs(n_rps))
            K_prop = self.rho * pow(D_prop, 5) * (
                    KQ_0 * abs(n_rps) * n_rps + (KQ_max - KQ_0) / Ja_max *
                    (Va / D_prop) * abs(n_rps))
            return X_prop, K_prop

        def reverse_thrust(n_rps):
            X_prop = self.rho * pow(D_prop, 4) * KT_0 * abs(n_rps) * n_rps
            K_prop = self.rho * pow(D_prop, 5) * KQ_0 * abs(n_rps) * n_rps
            return X_prop, K_prop

        X_prop, K_prop = lax.cond(n_rps > 0, forward_thrust, reverse_thrust, n_rps)



        # Rigi-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = m2c( self.MRB, nu_r)
        CA = m2c( self.MA, nu_r)

        # Nonlinear quadratic velocity terms in pitch and yaw (Munk moments)
        # are set to zero since only linear damping is used
        CA = CA.at[4,0].set(0)
        CA = CA.at[4,3].set(0)
        CA = CA.at[5,0].set(0)
        CA = CA.at[5,1].set(0)

        C = CRB + CA



        # Dissipative forces and moments
        D = jnp.diag(jnp.array([
            self.M[0][0] / self.T_surge * jnp.exp(-3 * U_r),
            self.M[1][1] / self.T_sway * jnp.exp(-3 * U_r),
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw * jnp.exp(-3 * U_r)
        ]))




        tau_liftdrag = forceLiftDrag( self.diam, self.S, self.CD_0, alpha, U_r)  # 阻力 升力
        tau_crossflow = crossFlowDrag( self.L, self.diam, self.diam, nu_r)  # 侧向水流（横流）作用下的阻力

        # Restoring forces and moments
        g = gvect( self.W, self.B, eta[4], eta[3], self.r_bg, self.r_bb)  # 计算一个潜在水中物体在任意点CO（通常是质心或者流体动力中心）的恢复力矢量。

        # Horizontal- and vertical-plane relative speed
        U_rh = jnp.sqrt(nu_r[0] ** 2 + nu_r[1] ** 2)
        U_rv = jnp.sqrt(nu_r[0] ** 2 + nu_r[2] ** 2)

        # Rudder and stern-plane drag
        X_r = -0.5 * self.rho * U_rh ** 2 * self.A_r * self.CL_delta_r * delta_r ** 2
        X_s = -0.5 * self.rho * U_rv ** 2 * self.A_s * self.CL_delta_s * delta_s ** 2


        # Rudder sway force
        Y_r = -0.5 * self.rho * U_rh ** 2 * self.A_r * self.CL_delta_r * delta_r

        # Stern-plane heave force
        Z_s = -0.5 * self.rho * U_rv ** 2 * self.A_s * self.CL_delta_s * delta_s


        # Generalized force vector
        tau = jnp.array([
            (1 - t_prop) * X_prop + X_r + X_s,
            Y_r,
            Z_s,
            K_prop / 10,  # scaled down by a factor of 10 to match exp. results
            self.x_s * Z_s,
            self.x_r * Y_r
        ], float)


        # AUV dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - jnp.matmul(C + D, nu_r) - g
        nu_dot = Dnu_c + jnp.matmul( self.Minv, tau_sum)  # 牛顿第二定律，求a


        # Actuator dynamics
        delta_r_dot = (delta_r_c - delta_r) / self.T_delta
        delta_s_dot = (delta_s_c - delta_s) / self.T_delta
        n_dot = (n_c - n) / self.T_n


        nu += sampleTime * nu_dot
        delta_r += sampleTime * delta_r_dot
        delta_s += sampleTime * delta_s_dot
        n += sampleTime * n_dot

        u_actual = jnp.array([delta_r, delta_s, n], float)

        return nu, u_actual, nu_dot

    @partial(jax.jit, static_argnums=(0))
    def reset(self):
        # self.t = 0  # initial simulation time
        # # Initial state vectors
        # self.eta = jnp.array([0, 0, 0, 0, 0, 0], float)  # position/attitude, user editable
        # self.nu = jnp.array([0, 0, 0, 0, 0, 0], float)  # velocity, defined by vehicle class
        # self.u_actual = jnp.array([0, 0, 0], float)  # actual inputs, defined by vehicle class

        # Initialization of table used to store the simulation data
        self.simData = jnp.empty([0, 2 * self.DOF + 2 * self.dimU], float)

    @partial(jax.jit, static_argnums=(0))
    def remus_solver(self, u_control, eta, nu, nu_c, u_actual, N=5, sampleTime=0.02):

        # for i in range(0, N ):
        #     # 这里返回的都是AUV坐标系计算的值
        #     [nu, u_actual, nu_dot] = self.dynamics(eta, nu, u_actual, u_control, sampleTime,nu_c)
        #     # 欧拉方法计算位置和速度,函数里使用2.53进行坐标系转化
        #     eta = attitudeEuler(eta, nu, sampleTime)

        def body_fun(i, carry):
            eta, nu, u_actual, nu_dot = carry
            # 这里返回的都是AUV坐标系计算的值
            nu, u_actual, nu_dot = self.dynamics(eta, nu, u_actual, u_control, sampleTime, nu_c)
            # 欧拉方法计算位置和速度,函数里使用2.53进行坐标系转化
            eta = attitudeEuler(eta, nu, sampleTime)
            return eta, nu, u_actual, nu_dot

        carry = (eta, nu, u_actual, jnp.zeros_like(nu))  # 初始时 nu_dot 设为与 nu 相同形状的零向量
        eta, nu, u_actual, nu_dot = lax.fori_loop(0, N, body_fun, carry)

        state_dot = jnp.zeros(12)
        # Kinematic Model
        state_dot = state_dot.at[:6].set(geom.J(eta).dot(nu))
        state_dot = state_dot.at[6:].set(nu_dot)
        return jnp.concatenate((eta, nu), axis=0), u_actual, state_dot

    @partial(jax.jit, static_argnums=(0))
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

        # if t > 100:
        #     delta_r = 0
        #
        # if t > 50:
        #     delta_s = 0

        delta_r = jnp.where(t > 100, 0, delta_r)
        delta_s = jnp.where(t > 50, 0, delta_s)

        u_control = jnp.array([delta_r, delta_s, n], float)

        return u_control

    @partial(jax.jit, static_argnums=(0))
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
        self.z_d = jnp.exp(-sampleTime * self.wn_d_z) * self.z_d \
                   + (1 - jnp.exp(-sampleTime * self.wn_d_z)) * z_ref

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

        u_control = jnp.array([delta_r, delta_s, n], float)

        return u_control
if __name__ == '__main__':
    
    # Test the Remus100 class
    remus = remus100()
    eta = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    nu = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    u_actual = jnp.array([0, 0, 0], dtype=jnp.float32)
    u_control = jnp.array([1.0, 0, 0], dtype=jnp.float32)
    nu_c = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    sampleTime = 0.02

    for i in range(2):
        print("round",i)
        # Test the dynamics function
        import time
        t = time.time()
        nu, u_actual, nu_dot = remus.dynamics(eta, nu, u_actual, u_control, sampleTime, nu_c)
        print(time.time() - t)
        print(nu, u_actual, nu_dot)


        # Test the remus_solver function
        t = time.time()
        eta_nu, u_actual, state_dot = remus.remus_solver(u_control, eta, nu, nu_c, u_actual)
        print(time.time() - t)
        print(eta_nu, u_actual, state_dot)


        # Test the stepInput function
        t = time.time()
        u_control = remus.stepInput(10)
        print(time.time() - t)
        print(u_control)

        # Test the depthHeadingAutopilot function
        t = time.time()
        u_control = remus.depthHeadingAutopilot(eta, nu, sampleTime)
        print(time.time() - t)
        print(u_control)