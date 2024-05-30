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

import gym
from gym.spaces import Discrete, Box
# import gymnasium as gym
# from gymnasium.spaces import Box
import matplotlib.pyplot as plt
import numpy as np
import wandb
from gym.utils import seeding

from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
from gym_dockauv.objects.current_v1 import Current
from gym_dockauv.objects.sensor import Radar
from gym_dockauv.objects.auvsim import AUVSim
from gym_dockauv.objects.shape import Sphere, Spheres, Capsule, intersec_dist_line_capsule_vectorized, \
    intersec_dist_lines_spheres_vectorized, collision_sphere_spheres, collision_capsule_sphere
import gym_dockauv.objects.shape as shape
import gym_dockauv.utils.geomutils as geom

from .docking3d_remus import BaseDocking3d_remus
# Set logger
logger = logging.getLogger(__name__)

"""
加一个随机位置
"""
class Timevaring_current_v2(BaseDocking3d_remus):
    """
    Set up and environment with multiple capsules as obstacles around the goal location (e.g. a pear or oil rig)
    Add water current to it
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__(env_config)
        if "goal_point" in env_config:
            self.goal_point_ = env_config["goal_point"]
        else:
            raise NotImplementedError
        if "start_point" in env_config:
            self.start_point_ = env_config["start_point"]

        if "random_position" in env_config and env_config["random_position"]:
            self.random_position = env_config["random_position"]

            self.start_goal_radius = env_config["start_goal_radius"]

            self.n_obs_without_radar = self.n_obs_without_radar  + 3
            # self.n_observations = self.n_obs_without_radar + self.radar.n_rays_reduced
            self.n_observations = self.n_obs_without_radar + 0
            obs_low = -np.ones(self.n_observations)
            obs_low[0] = 0
            obs_low[self.n_obs_without_radar:] = 0
            self.observation_space = Box(low=obs_low,
                                         high=np.ones(self.n_observations),
                                         dtype=np.float32)
            self.observation = np.zeros(self.n_observations)

        else:
            self.random_position = None

        if "random_attitude" in env_config and env_config["random_attitude"]:
            self.random_attitude = env_config["random_attitude"]
        else:
            self.random_attitude = None


    def generate_environment(self):
        """
        Set up an environment after each reset call, can be used to in multiple environments to make multiple scenarios
        """
        # Same setup as before:
        # super().generate_environment()
        # Water current
        curr_angle = (np.random.random(2) - 0.5) * 2 * np.array([np.pi / 2, np.pi])  # Water current direction
        self.current = Current(mu=0.005, V_min=0.5, V_max=0.5, Vc_init=0.5,
                               alpha_init=curr_angle[0], beta_init=curr_angle[1], white_noise_std=0.0,
                               step_size=self.auv.step_size,current_on=self.current_on,config=self.config)
        self.nu_c = self.current(self.auv.attitude, position=self.auv.position)

        if self.random_position is not None:
            random_list_1 = list((np.random.random(3) - 0.5) * 2 * self.start_goal_radius)
            self.goal_location = [self.goal_point_[i] + random_list_1[i] for i in range(len(self.goal_point_))]
            random_list_2 = list((np.random.random(3) - 0.5) * 2*self.random_position)
            self.auv.position = [self.start_point_[i] + random_list_2[i] for i in range(len(self.start_point_))]
            self.start_location = copy.deepcopy(self.auv.position)
        else:
            self.goal_location = self.goal_point_
            self.auv.position = self.start_point_
            self.start_location = copy.deepcopy(self.auv.position)

        # Attitude
        if self.random_attitude:
            self.auv.attitude = self.generate_random_att(max_att_factor=0.7)
        else:
            self.auv.attitude = np.array([0, 0, 0.49 * np.pi])

        ###########
        CAPSULE_RADIUS = 1.0
        CAPSULE_HEIGHT = 4.0
        # Obstacles (only the capsule at goal location):
        cap = Capsule(position=np.array([0.0, 0.0, 0.0]),
                      radius=CAPSULE_RADIUS,
                      vec_top=np.array([0.0, 0.0, -CAPSULE_HEIGHT / 2.0]))
        self.capsules = [cap]
        self.obstacles = [*self.capsules]
        # Get vector pointing from goal location to capsule:
        vec = shape.vec_line_point(self.goal_location, cap.vec_top, cap.vec_bot)
        # Calculate heading at goal
        self.heading_goal_reached = geom.ssa(np.arctan2(vec[1], vec[0]))
        ###########

        ######
        CAPSULE_OBSTACLES_RADIUS = 2.0
        CAPSULE_OBSTACLES_HEIGHT = 2 * self.max_dist_from_goal
        CAPSULE_DISTANCE_FROM_CENTER = 6
        NUMBER_OF_CAPSULES = 1
        # Generate new capsules
        new_capsules = []
        theta = np.random.rand() * 2 * np.pi
        for i in range(NUMBER_OF_CAPSULES):
            # x = np.cos(theta) * CAPSULE_DISTANCE_FROM_CENTER
            # y = np.sin(theta) * CAPSULE_DISTANCE_FROM_CENTER
            x = -130.
            y = 0.
            # Create new capsule and append to list
            new_capsules.append(
                Capsule(position=np.array([x, y, 0.0]),
                        radius=CAPSULE_OBSTACLES_RADIUS,
                        vec_top=np.array([x, y, -CAPSULE_OBSTACLES_HEIGHT / 2.0]))
            )
        # Add Obstacles:
        self.capsules.extend(new_capsules)
        self.obstacles.extend(new_capsules)
        ######
