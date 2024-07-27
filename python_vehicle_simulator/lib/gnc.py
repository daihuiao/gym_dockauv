#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNC functions. 

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

# import numpy as  jnp
from jax import numpy as  jnp
import math

#------------------------------------------------------------------------------

def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle +  jnp.pi) % (2 *  jnp.pi) -  jnp.pi

    return angle

#------------------------------------------------------------------------------

def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min

    return x

#------------------------------------------------------------------------------

def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b. 
    """

    S =  jnp.array([
        [ 0, -a[2], a[1] ],
        [ a[2],   0,     -a[0] ],
        [-a[1],   a[0],   0 ]  ])

    return S


#------------------------------------------------------------------------------

def Hmtrx(r):
    """
    H = Hmtrx(r) computes the 6x6 system transformation matrix
    H = [eye(3)     S'
         zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

    If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
    CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
    force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 
    """

    H =  jnp.identity(6,float)
    H = H.at[0:3, 3:6].set(Smtrx(r).T)

    return H

#------------------------------------------------------------------------------

def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    2.31
    2.53
    """

    cphi =  jnp.cos(phi)
    sphi =  jnp.sin(phi)
    cth  =  jnp.cos(theta)
    sth  =  jnp.sin(theta)
    cpsi =  jnp.cos(psi)
    spsi =  jnp.sin(psi)

    R =  jnp.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

#------------------------------------------------------------------------------

def Tzyx(phi,theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """

    cphi =  jnp.cos(phi)
    sphi =  jnp.sin(phi)
    cth  =  jnp.cos(theta)
    sth  =  jnp.sin(theta)

    try:
        T =  jnp.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])

    except ZeroDivisionError:
        print ("Tzyx is singular for theta = +-90 degrees." )

    return T

#------------------------------------------------------------------------------

def attitudeEuler(eta,nu,sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
    position/Euler angles eta[k+1]
    2.53
    """

    p_dot   =  jnp.matmul( Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
    v_dot   =  jnp.matmul( Tzyx(eta[3], eta[4]), nu[3:6] )

    # Forward Euler integration
    eta = eta.at[0:3].set(eta[0:3] + sampleTime * p_dot)
    eta = eta.at[3:6].set(eta[3:6] + sampleTime * v_dot)

    return eta


#------------------------------------------------------------------------------

def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3) 3.51
    """
    """
    质量矩阵对称化：M = 0.5 * (M + M.T)。这一步确保质量矩阵 M 是对称的，这在物理和数学上是必要的。
    6自由度模型：如果使用的是6自由度模型，代码将质量矩阵分成四个部分：M11, M12, M21, M22。然后，
    它分别计算这些矩阵和速度向量 nu1、nu2 的乘积，得到 dt_dnu1 和 dt_dnu2。最后，
    C 矩阵通过组合这些乘积和使用 Smtrx 函数（生成斜对称矩阵）构建而成。
    首先计算了由质量矩阵的子矩阵和速度向量 nu1、nu2 组成的向量 dt_dnu1 和 dt_dnu2。这些向量表示由于船舶的速度变化而产生的动量变化率。

    接着，构建科里奥利和离心力矩阵 C。这个矩阵由两部分组成：
    
    C[0:3,3:6] 和 C[3:6,0:3] 都是使用 -Smtrx(dt_dnu1) 计算得到的，代表了速度变化对旋转运动的影响。
    C[3:6,3:6] 使用 -Smtrx(dt_dnu2) 计算，代表了旋转速度变化对自身产生的影响。
    Smtrx 函数用于生成一个斜对称矩阵，这是计算科里奥利和离心力矩阵的重要一步。在物理上，这些矩阵代表了由于船舶的旋转和线性运动变化而产生的附加力和力矩。"""

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model

        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6]
        M21 = M12.T
        M22 = M[3:6,3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 =  jnp.matmul(M11,nu1) +  jnp.matmul(M12,nu2)
        dt_dnu2 =  jnp.matmul(M21,nu1) +  jnp.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C =  jnp.zeros( (6,6) )
        C = C.at[0:3,3:6].set(-Smtrx(dt_dnu1))
        C = C.at[3:6,0:3].set(-Smtrx(dt_dnu1))
        C = C.at[3:6,3:6].set(-Smtrx(dt_dnu2))

    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
        C =  jnp.zeros( (3,3) )
        C = C.at[0,2].set(-M[1,1] * nu[1] - M[1,2] * nu[2])
        C = C.at[1,2].set(M[0,0] * nu[0])
        C = C.at[2,0].set(-C[0,2])
        C = C.at[2,1].set(-C[1,2])

    return C

#------------------------------------------------------------------------------

def Hoerner(B,T):
    """
    CY_2D = Hoerner(B,T)
    Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
    B and draft T.The data is digitized and interpolation is used to compute 
    other data point than those in the table
    """

    # DATA = [B/2T  C_D]
    DATA1 =  jnp.array([
        0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
        0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031
        ])
    DATA2 =  jnp.array([
        1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
        0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593
        ])

    CY_2D =  jnp.interp( B / (2 * T), DATA1, DATA2 )

    return CY_2D

#------------------------------------------------------------------------------

def crossFlowDrag(L,B,T,nu_r):
    """
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    integrals for a marine craft using strip theory. 使用带状理论计算海洋飞船的跨流阻力积分

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow

    """
    """    这个函数 crossFlowDrag 计算海洋船舶在侧向水流（横流）作用下的阻力。它使用带状理论，将船体分成多个小段（这里是20段），
        并且计算每一段因水流相对船体的横向速度而产生的力和力矩。这个计算涉及船体的长度 L，宽度 B，吃水深度 T，
        以及相对速度向量 nu_r。计算结果是一个力矩向量 tau_crossflow，它表示由于横流阻力而作用在船体上的侧向力和偏航力矩。
    通过这个函数，可以帮助模拟船舶在侧向水流影响下的动态行为。"""
    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20
    Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve 基于Hoerner的曲线的2D阻力系数

    Yh = 0
    Nh = 0
    xL = -L/2

    for i in range(0,n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[5]               # yaw rate
        Ucf = abs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx

    tau_crossflow =  jnp.array([0, Yh, 0, 0, 0, Nh],float)

    return tau_crossflow

#------------------------------------------------------------------------------

def forceLiftDrag(b,S,CD_0,alpha,U_r):
    """
    tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
    lift and drag forces of a submerged "wing profile" for varying angle of
    attack (Beard and McLain 2012). Application:
    
      M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag
    
    I jnputs:
        b:     wing span (m)
        S:     wing area (m^2)
        CD_0:  parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
        alpha: angle of attack, scalar or vector (rad)
        U_r:   relative speed (m/s)

    Returns:
        tau_liftdrag:  6x1 generalized force vector
    """
    """
    这段代码计算了水下“机翼型”物体的升力和阻力。这对于了解和模拟水下航行器或类似物体在水中运动非常重要。详细解释如下：

    函数定义和输入：forceLiftDrag 函数接受机翼型物体的跨度（b）、面积（S）、零攻角阻力系数（CD_0）、攻角（alpha）和相对速度（U_r）作为输入。
    
    计算升力和阻力系数：函数内部，首先通过 coeffLiftDrag 函数计算升力系数（CL）和阻力系数（CD），这两个系数是攻角（alpha）的函数。
    
    阻力和升力的计算：接着，使用计算出的阻力和升力系数，结合相对速度（U_r）和流体密度（rho），计算出阻力（F_drag）和升力（F_lift）。
    
    力转换：最后，将这些力从流体坐标系转换到物体本体坐标系，并组合成一个广义力向量 tau_liftdrag。这个向量包含了由于升力和阻力产生的力和力矩。"""
    # constants
    rho = 1026

    def coeffLiftDrag(b,S,CD_0,alpha,sigma):

        """
        [CL,CD] = coeffLiftDrag(b,S,CD_0,alpha,sigma) computes the hydrodynamic 
        lift CL(alpha) and drag CD(alpha) coefficients as a function of alpha
        (angle of attack) of a submerged "wing profile" (Beard and McLain 2012)

        CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
        CL(alpha) = CL_0 + CL_alpha * alpha
  
        where CD_p is the parasitic drag (profile drag of wing, friction and
        pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle 
        of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the  
        Oswald efficiency number. For lift it is assumed that
  
        CL_0 = 0
        CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );
  
        implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
        high angles of attack the linear lift model can be blended with a
        nonlinear model to describe stall
  
        CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
            sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

        where 0 <= sigma <= 1 is a blending parameter. 
        
        I jnputs:
            b:       wing span (m)
            S:       wing area (m^2)
            CD_0:    parasitic drag (alpha = 0), typically 0.1-0.2 for a 
                     streamlined body
            alpha:   angle of attack, scalar or vector (rad)
            sigma:   blending parameter between 0 and 1, use sigma = 0 f
                     or linear lift 
            display: use 1 to plot CD and CL (optionally)
        
        Returns:
            CL: lift coefficient as a function of alpha   
            CD: drag coefficient as a function of alpha   

        Example:
            Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
            CD_0 = 0.3
            
            alpha = 0.1 * pi/180
            [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
        """
        """这个 coeffLiftDrag 函数计算水下物体（如水翼型船体）的升力系数 (CL) 和阻力系数 (CD)。这些系数是攻角 (alpha) 的函数，用于描述物体在水中的流体动力学特性。
        
        输入参数：函数接受翼展 (b)、翼面积 (S)、零攻角阻力系数 (CD_0)、攻角 (alpha) 和混合参数 (sigma)。
        
        效率和纵横比：计算了奥斯瓦尔德效率数 (e) 和翼的纵横比 (AR)。
        
        线性升力模型：使用简化公式计算线性升力系数 (CL_alpha)，然后计算 CL。
        
        阻力系数：计算 CD，包括寄生阻力和由升力引起的诱导阻力。
        
        非线性升力模型：在高攻角下，使用非线性模型对 CL 进行调整。"""
        e = 0.7             # Oswald efficiency number
        AR = b**2 / S       # wing aspect ratio

        # linear lift
        CL_alpha =  jnp.pi * AR / ( 1 +  jnp.sqrt(1 + (AR/2)**2) )
        CL = CL_alpha * alpha

        # parasitic and induced drag
        CD = CD_0 + CL**2 / ( jnp.pi * e * AR)

        # nonlinear lift (blending function)
        CL = (1-sigma) * CL + sigma * 2 *  jnp.sign(alpha) \
            *  jnp.sin(alpha)**2 *  jnp.cos(alpha)

        return CL, CD


    [CL, CD] = coeffLiftDrag(b,S,CD_0,alpha,0) #8.89 8.90

    F_drag = 1/2 * rho * U_r**2 * S * CD    # drag force 阻力
    F_lift = 1/2 * rho * U_r**2 * S * CL    # lift force 升力

    # transform from FLOW axes to BODY axes using angle of attack
    tau_liftdrag =  jnp.array([
         jnp.cos(alpha) * (-F_drag) -  jnp.sin(alpha) * (-F_lift),
        0,
         jnp.sin(alpha) * (-F_drag) +  jnp.cos(alpha) * (-F_lift),
        0,
        0,
        0 ])

    return tau_liftdrag

#------------------------------------------------------------------------------

def gvect(W,B,theta,phi,r_bg,r_bb):
    """
    g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
    forces about an arbitrarily point CO for a submerged body. 
    
    I jnputs:
        W, B: weight and buoyancy (kg)
        phi,theta: roll and pitch angles (rad)
        r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
        r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)
        
    Returns:
        g: 6x1 vector of restoring forces about CO
        这个函数 gvect 用于计算一个潜在水中物体在任意点CO（通常是质心或者流体动力中心）的恢复力矢量。
        它的输入参数包括物体的重量 W 和浮力 B，俯仰角 θ 和横滚角 ϕ，以及重心 CG 和浮心 CB 相对于CO的位置 r bg  和 r bb 。
        函数计算六个分量，前三个分量对应于物体在三个方向上的力（即X、Y和Z方向），而后三个分量对应于围绕三个轴的力矩
        （即绕X轴的滚转力矩，绕Y轴的俯仰力矩，以及绕Z轴的偏航力矩）。这个恢复力矢量是在均衡位置附近小扰动的情况下，
        使得物体恢复到平衡位置的力和力矩的线性近似。
    """

    sth  =  jnp.sin(theta)
    cth  =  jnp.cos(theta)
    sphi =  jnp.sin(phi)
    cphi =  jnp.cos(phi)

    g =  jnp.array([
        (W-B) * sth,
        -(W-B) * cth * sphi,
        -(W-B) * cth * cphi,
        -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
        (r_bg[2]*W-r_bb[2]*B) * sth         + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
        -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth
        ])

    return g


