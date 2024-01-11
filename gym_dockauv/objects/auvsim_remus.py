import copy
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
from python_vehicle_simulator.lib.gnc import ssa, Rzyx, Tzyx

from .statespace import StateSpace
from ..utils import geomutils as geom
from ..utils.lowpassfilter import LowPassFilter
from ..utils.odesolver45 import odesolver45
import time
from python_vehicle_simulator.vehicles.remus100 import remus100


class AUVSim_remus(StateSpace, ABC):
    r"""
    This class serves as an interface base class for the AUV instance. It handles the input and output to deal with
    the state space equations, save the state of the vehicle and represent the state of the vehicle in the
    Simulation. Below you find the most important definitions:

    .. math::

        \boldsymbol{\eta} = \begin{bmatrix}
            x & y & z & \phi & \theta & \psi
        \end{bmatrix}^T

        \boldsymbol{\nu} = \begin{bmatrix}
            u & v & w & p & q & r
        \end{bmatrix}^T

    .. note:: Assumptions:

        - This implementation always assumes a low pass filter over the input for simplification
          (can always be overwritten, in case you have your own implementation from input to external forces)
        - Action space for the DRL agent or NN is in the interval of [-1, 1]

    """

    def __init__(self):
        super().__init__()
        # These values should be overwritten by a config for a run scenario, default here
        self.state = np.hstack([np.zeros((6,)), np.zeros((6,))])
        self._state_dot = np.hstack([np.zeros((6,)), np.zeros((6,))])
        self.lowpassfilter = LowPassFilter(T1=0.2, sample_time=1)  # todo dai
        self.step_size = 1  # This would also automatically update lowpassfilter due to setter
        # TODO: Make step_size clear where to initiate and how
        self.safety_radius = 1

        # Standard initialization
        # Make B dependent implementation of input vector u:
        self._u = None
        self.remus = remus100()

    def __setattr__(self, name, value):
        # Overwrite Setter, to automatically update low-pass filter too
        if name == 'step_size':
            self.lowpassfilter.sample_time = value
        super().__setattr__(name, value)

    def reset(self):
        """
        Function to reset the simulated vehicle entirely

        .. note:: u can be set to None, since if it is called again without new initialization, it sets itself to
            zero automatically
        """
        self.remus.reset()
        self.state = np.hstack([np.zeros((6,)), np.zeros((6,))])
        self.last_state = copy.deepcopy(self.state)
        self._state_dot = np.hstack([np.zeros((6,)), np.zeros((6,))])
        self.u = None
        self.u_actual = np.array([0, 0, 0], float)  # actual inputs, defined by vehicle class

    def unnormalize_input(self, norm_input: np.ndarray) -> np.ndarray:
        """
        Assumes action space between -1 and 1 and converts in linear to the bounded real action space of the vehicle

        :param norm_input: input from e.g. a neural network (nn) array ax1
        :return: arrax ax1
        """
        input_c = np.clip(norm_input, -1, 1)
        return self.u_bound[:, 0] + (self.u_bound[:, 1] - self.u_bound[:, 0]) * (input_c + 1) / 2

    def step(self, action: np.ndarray, nu_c: np.ndarray) -> None:
        r"""
        Apply one simulation step with Un-normalize action input from DRL or NN and apply low-pass filter

        :param action: normalized input by DRL agent or NN
        :param nu_c: water current speed vector
        :return: None
        """
        # Un-normalize action input from outside and apply low-pass filter to changes
        # self.u = self.lowpassfilter.apply_lowpass(self.unnormalize_input(action), self.u)
        # self.u = self.unnormalize_input(action)
        self.u = action  # bug fixed ,why? 峨峨峨,我知道了，他原来的代码就有问题，sb3本来就会做一个去皈依化，而他可能之前用的其他算法库。
        self._sim(nu_c)

    def _sim(self, nu_c: np.ndarray) -> None:
        """
        Simulation step with own defined odesolver45. It has been checked against the python scipy.integrate module
        and achieves the same result in less time due to less overhead

        :param nu_c: water current speed vector
        :return: None
        """
        if False:
            # Perform ODE simulation step
            # t = time.time()
            self.state1, q = odesolver45(f=self.state_dot, t=0, y=self.state, h=self.step_size, **{"nu_c": nu_c})
            # print("ODE solver time: ", time.time() - t)

            # Alternative with official python package - Note: There are a lot of ways to hack a constant step size,
            # none of them are beautiful
            # t = time.time()
            # res = solve_ivp(fun=self.state_dot, t_span=[0, self.step_size],
            #                 y0=self.state, t_eval=[self.step_size], method='RK45', args=(nu_c,))
            # print("Scipy solver time: ", time.time() - t)
            #
            # self.state2 = res.y.flatten()

            # Convert angle in applicable range
            self.state[3:6] = geom.ssa(self.state[3:6])  # 将欧拉角限制在[-pi, pi]之间
            self._state_dot = self.state_dot(0, self.state, nu_c)  # Save the speed here
        else:
            self.state, self.u_actual, self._state_dot = self.remus.remus_solver(self.u, eta=self.state[:6],
                                                                                 nu=self.state[6:], nu_c=nu_c,
                                                                                 u_actual=self.u_actual)
            # distance_moved = np.linalg.norm(self.state[:3] - self.last_state[:3])
            # self.last_state = copy.deepcopy(self.state)

            # Convert angle in applicable range
            self.state[3:6] = geom.ssa(self.state[3:6])  # 将欧拉角限制在[-pi, pi]之间
            # self._state_dot = self.state_dot(0, self.state, nu_c)  # Save the speed here

    def state_dot(self, t, state, nu_c: np.ndarray) -> np.ndarray:
        r"""
        The right-hand side (RHS) of the 12 ODEs governing the AUV dynamics. Including:

        .. math::

            \boldsymbol{\dot{\eta}} = \boldsymbol{J}_{\Theta}(\boldsymbol{\eta}) \boldsymbol{\nu}
            \iff
            \begin{bmatrix}
                \boldsymbol{\dot{p}} \\
                \boldsymbol{\dot{\Theta}_{nb}}
            \end{bmatrix}
            =
            \begin{bmatrix}
                \boldsymbol{R}_b^n(\boldsymbol{\Theta}_{nb}) & 0\\
                0 & \boldsymbol{T}_{\Theta}(\boldsymbol{\Theta}_{nb})
            \end{bmatrix}
            \begin{bmatrix}
                \boldsymbol{v}^b \\
                \boldsymbol{\omega}^b
            \end{bmatrix}

        .. math::

            \underbrace{\boldsymbol{M}_{RB} \boldsymbol{\dot{\nu}}
            + \boldsymbol{C}_{RB}(\boldsymbol{\nu})\boldsymbol{\nu}}_{\text{rigid-body terms}}
            + \underbrace{\boldsymbol{M}_A \boldsymbol{\dot{\nu}}_r + C_A(\boldsymbol{\nu}_r) \boldsymbol{\nu}_r
            + D(\boldsymbol{\nu}_r) \boldsymbol{\nu}_r}_{\text{hydrodynamic terms}}
            + \underbrace{g(\eta)}_{\text{hydrostatic terms}} = \boldsymbol{\tau}

        :param t: Dummy variable used for most ode solvers
        :param state: state vector 12x1 of vehicle
        :param nu_c: water current nu state vector
        :return: array 12x1
        """
        eta = state[:6]
        nu_r = state[6:]

        state_dot = np.zeros(12)

        # Kinematic Model
        state_dot[:6] = geom.J(eta).dot(nu_r + nu_c)

        # Kinetic Model
        state_dot[6:] = self.M_inv.dot(
            self.B(nu_r).dot(self.u)
            - self.D(nu_r).dot(nu_r)
            - self.C(nu_r).dot(nu_r)
            - self.G(eta))

        return state_dot

    @property
    def position(self):
        r"""
        Returns the position of the AUV in NED coordinates x, y, z.

        .. math::

            \boldsymbol{p} = \begin{bmatrix}
                x & y & z
            \end{bmatrix}^T

        """
        return self.state[0:3]

    @position.setter
    def position(self, value):
        """Setter for the position into state"""
        self.state[0:3] = value.copy()

    @property
    def attitude(self):
        r"""
        Returns the attitude (euler angles) of the AUV wrt. to NED coordinates roll, pitch, yaw.

        .. math::

            \boldsymbol{\Theta} = \begin{bmatrix}
                \phi & \theta & \psi
            \end{bmatrix}^T
        """
        return self.state[3:6]

    @attitude.setter
    def attitude(self, value):
        """Setter for the attitude into state"""
        self.state[3:6] = value.copy()

    @property
    def eta(self):
        r"""
        Returns the eta, which includes the position and attitude

        .. math::

            \boldsymbol{\eta} = \begin{bmatrix}
                x & y & z & \phi & \theta & \psi
            \end{bmatrix}^T
        """
        return self.state[0:6]

    @property
    def relative_velocity(self):
        r"""
        Returns the surge, sway and heave velocity of the AUV.

        .. math::

            \boldsymbol{v} = \begin{bmatrix}
                u & v & w
            \end{bmatrix}^T
        """
        return self.state[6:9]

    @property
    def ned_velocity(self):#todo dai: 这个函数不保熟，小心食用
        r"""
        Returns the NED velocity of the AUV.

        .. math::

            \boldsymbol{v} = \begin{bmatrix}
                u & v & w
            \end{bmatrix}^T
        """
        velocity = np.matmul(Rzyx(self.eta[3], self.eta[4], self.eta[5]), self.state[6:9])
        return velocity

    @property
    def relative_speed(self):
        """
        Returns the total speed of the AUV.
        """
        return np.linalg.norm(self.relative_velocity)

    @property
    def angular_velocity(self):
        r"""
        Returns the rate of rotation about the Body frame.

        .. math::

            \boldsymbol{\omega} = \begin{bmatrix}
                p & q & r
            \end{bmatrix}^T
        """
        return self.state[9:12]

    @property
    def position_dot(self):
        r"""
        Returns :math:`\boldsymbol{\dot{p}}`
        """
        return self._state_dot[0:3]

    @property
    def euler_dot(self):
        r"""
        Returns :math:`\boldsymbol{\dot{Theta}}`
        """
        return self._state_dot[3:6]

    @property
    def chi(self):
        r"""
        Returns the azimuth angle :math:`\chi` .
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(E_dot, N_dot)

    @property
    def upsilon(self):
        r"""
        Returns the inclination (path) angle :math:`\Upsilon`
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(-D_dot, np.sqrt(N_dot ** 2 + E_dot ** 2))

    @property
    def u(self) -> np.ndarray:
        # After u_bound is known in child class, make sure it is initialized to the right size and as zero
        if self._u is None:
            self._u = np.zeros(self.u_bound.shape[0])
        return self._u

    @u.setter
    def u(self, value):
        self._u = value
