"""Classic control environments."""

from gymnax.environments.classic_control import acrobot
from gymnax.environments.classic_control import cartpole
from gymnax.environments.classic_control import continuous_mountain_car
from gymnax.environments.classic_control import mountain_car
from gymnax.environments.classic_control import pendulum
from gymnax.environments.classic_control import remus_v1


Acrobot = acrobot.Acrobot
CartPole = cartpole.CartPole
ContinuousMountainCar = continuous_mountain_car.ContinuousMountainCar
MountainCar = mountain_car.MountainCar
Pendulum = pendulum.Pendulum
Remus_v1 = remus_v1.remus_v1


__all__ = [
    "Pendulum",
    "CartPole",
    "MountainCar",
    "ContinuousMountainCar",
    "Acrobot",
    "Remus_v1",
]
