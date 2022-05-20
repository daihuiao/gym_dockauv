import datetime
import importlib
import logging
import os
import pprint
import time
from timeit import default_timer as timer
from typing import Tuple, Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from gym_dockauv.config.env_config import BASE_CONFIG
from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.utils.plotutils import EpisodeAnimation
import gym_dockauv.objects.shape as shape

# TODO: Think about making this a base class for further environments with generate environment functions!
# TODO: Save animation option
# TODO: Water current, radar sensors, obstacles (so far only capsules are supported)

# Set logger
logger = logging.getLogger(__name__)


class Docking3d(gym.Env):
    """
    Base Class for the docking environment, will also be registered with gym. However, the configs for the
    environment are found at gym_dockauv/config

    .. note:: Adding a reward or a done condition with reward needs to take the following steps:
        - Add reward to the self.last_reward_arr in the reward step
        - Add a factor to it in the config file
        - Update number of self.n_rewards in __init__()
        - Update the list self.meta_data_reward in __init__()
        - Update the index of self.meta_data_done in __init__() if necessary
        - Update the doc for the reward_step() function (and of done())
    """

    def __init__(self, env_config: dict = BASE_CONFIG):
        super().__init__()
        # Basic config for logger
        self.config = env_config
        self.title = self.config["title"]
        self.save_path_folder = self.config["save_path_folder"]
        self.log_level = self.config["log_level"]
        self.verbose = self.config["verbose"]
        os.makedirs(self.save_path_folder, exist_ok=True)
        # Initialize logger
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        logging.basicConfig(level=self.log_level,
                            filename=os.path.join(self.save_path_folder, f"{utc_str}__{self.title}.log"),
                            format='[%(asctime)s] [%(levelname)s] [%(module)s] - [%(funcName)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                            )
        # Check if logging statement are supposed to go to std output
        if self.verbose:
            logging.getLogger().addHandler(logging.StreamHandler())
        logging.Formatter.converter = time.gmtime  # Make sure to use UTC time in logging timestamps

        logger.info('---------- Docking3d Gym Logger ----------')
        logger.info('---------- ' + utc_str + ' ----------')
        logger.info('---------- Initialize environment ----------')
        logger.info('Gym environment settings: \n ' + pprint.pformat(env_config))

        # Dynamically load class of vehicle and instantiate it (available vehicles under gym_dockauv/objects/vehicles)
        AUV = getattr(importlib.import_module("gym_dockauv.objects.vehicles." + self.config["vehicle"]),
                      self.config["vehicle"])
        self.auv = AUV()

        # Set step size for vehicle
        self.auv.step_size = self.config["t_step_size"]

        # Set the action and observation space
        self.n_observations = 13
        self.action_space = gym.spaces.Box(low=self.auv.u_bound[:, 0],
                                           high=self.auv.u_bound[:, 1],
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.n_observations),
                                                high=np.ones(self.n_observations),
                                                dtype=np.float32)
        self.observation = np.zeros(self.n_observations)

        # General simulation variables:
        self.t_total_steps = 0  # Number of total timesteps run so far in this environment
        self.t_steps = 0  # Number of steps in this episode
        self.t_step_size = self.config["t_step_size"]
        self.episode = 0  # Current episode
        self.max_timesteps = self.config["max_timesteps"]
        self.interval_datastorage = self.config["interval_datastorage"]
        self.info = {}  # This will contain general simulation info

        # Declaring further own attributes
        self.obstacles = []
        self.goal_reached = False  # Bool to check of goal is reached at the end of an episode
        self.collision = False  # Bool to indicate of vehicle has collided

        # Rewards
        self.n_rewards = 8
        self.last_reward = 0  # Last reward
        self.last_reward_arr = np.zeros(self.n_rewards)  # This should reflect the dimension of the rewards parts
        self.cumulative_reward = 0  # Current cumulative reward of agent
        self.cum_reward_arr = np.zeros(self.n_rewards)
        self.conditions = None  # Boolean array to see which conditions are true
        # Description for the meta data
        self.meta_data_reward = [
            "Distance_goal",
            "Attitude",
            "time_step",
            "action",
            "Done-Goal_reached",
            "Done-out_pos",
            "Done-out_att",
            "Done-max_t"
        ]
        self.reward_factors = self.config["reward_factors"]
        self.action_reward_factors = self.config["action_reward_factors"]

        # Initialize Done condition and related stuff for the done condition
        self.done = False
        self.meta_data_done = self.meta_data_reward[4:]
        self.goal_location = self.config["goal_location"]
        self.max_dist_from_goal = self.config["max_dist_from_goal"]
        self.max_attitude = self.config["max_attitude"]

        # Water current TODO
        self.nu_c = np.zeros(6)

        # Save and display simulation time
        self.start_time_sim = timer()

        # Episode Data storage
        self.episode_data_storage = None

        # Full data storage:
        self.full_data_storage = FullDataStorage()
        self.full_data_storage.set_up_episode_storage(env=self, path_folder=self.save_path_folder, title=self.title)

        # Animation variables
        self.episode_animation = None
        self.ax = None

        logger.info('---------- Initialization of environment complete ---------- \n')
        logger.info('---------- Rewards function description ----------')
        logger.info(self.reward_step.__doc__)

    def reset(self, seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,
              ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        From Base Class:

        Resets the environment to an initial state and returns an initial
        observation.

        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and `reset` is called with `seed=None`,
        the RNG should not be reset.
        Moreover, `reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        .. note:: Options parameter not used yet
        """
        # In case any windows were open from matplotlib or animation
        if self.episode_animation:
            plt.close(self.episode_animation.fig)  # TODO: Window stays open, prob due to Gym
            self.episode_animation = None
            self.ax = None

        # Save info to return in the end
        return_info_dict = self.info.copy()

        # Check if we should save a datastorage item
        if self.episode_data_storage and (self.episode % self.interval_datastorage == 0 or self.episode == 1):
            self.episode_data_storage.save()
        self.episode_data_storage = None

        # Update Full data storage:
        if self.episode != 0:
            self.full_data_storage.update()

        # ---------- General reset from here on -----------
        self.auv.reset()
        self.t_steps = 0
        self.goal_reached = False
        self.collision = False
        self.info = {}

        # Reset observation, cum_reward, done, info
        self.observation = np.zeros(self.n_observations, dtype=np.float32)
        self.last_reward = 0
        self.cumulative_reward = 0
        self.last_reward_arr = np.zeros(self.n_rewards)
        self.cum_reward_arr = np.zeros(self.n_rewards)
        self.done = False
        self.conditions = None

        # Update the seed:
        # TODO: Check if this makes all seeds same (e.g. for water current!!) or works in general
        # Comment Thomas: maybe need to fix at 2-3 other places
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Update episode number
        self.episode += 1

        # ----------- Init for new environment from here on -----------
        # Generate environment
        self.generate_environment()

        # Save whole episode data if interval is met, or we need it for the renderer
        if self.episode % self.interval_datastorage == 0 or self.episode == 1:
            self.init_episode_storage()
        else:
            self.episode_data_storage = None

        # Log the episode
        logger.info("Environment reset call: \n" + pprint.pformat(return_info_dict))

        # Return info if wanted
        if return_info:
            return self.observation, return_info_dict
        return self.observation

    def generate_environment(self):
        """
        Setup a environment after each reset call
        """
        # TODO Think about how this should be done in future simulations
        # Position
        rnd_arr_pos = (np.random.random(3) - 0.5)
        self.auv.position = rnd_arr_pos * (6 / np.linalg.norm(rnd_arr_pos))
        # Attitude
        rnd_arr_attitude = (np.random.random(3) - 0.5)
        att_factor = np.array([self.max_attitude*0.7, self.max_attitude*0.7, np.pi])  # Spawn at xx% of max attitude
        self.auv.attitude = rnd_arr_attitude * att_factor  # Spawn with random attitude

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Simulate current TODO

        # Update AUV dynamics
        self.auv.step(action, self.nu_c)

        # Check collision TODO

        # Update data storage if active
        if self.episode_data_storage:
            self.episode_data_storage.update(self.nu_c, self.cum_reward_arr, self.last_reward_arr)

        # Update visualization if active
        if self.episode_animation:
            self.render()

        # Determine if simulation is done, this also updates self.last_reward
        self.done, cond_idx = self.is_done()

        # Calculate rewards
        self.last_reward = self.reward_step(action)
        self.cumulative_reward += self.last_reward

        # Make next observation TODO
        self.observation = self.observe()

        # Save sim time info
        self.t_total_steps += 1
        self.t_steps += 1

        # Update info dict
        self.info = {"episode_number": self.episode,  # Need to be episode number, because episode is used by sb3
                     "t_step": self.t_steps,
                     "t_total_steps": self.t_total_steps,
                     "cumulative_reward": self.cumulative_reward,
                     "last_reward": self.last_reward,
                     "done": self.done,
                     "conditions_true": cond_idx,
                     "conditions_true_info": [self.meta_data_done[i] for i in cond_idx],
                     "collision": self.collision,
                     "goal_reached": self.goal_reached,
                     "simulation_time": timer() - self.start_time_sim}

        return self.observation, self.last_reward, self.done, self.info

    def observe(self) -> np.ndarray:
        diff = self.goal_location - self.auv.position
        obs = np.zeros(self.n_observations, dtype=np.float32)
        obs[0] = np.clip(diff[0] / self.max_dist_from_goal, -1, 1)  # TODO: Position difference, with max_dist, incorrect
        obs[1] = np.clip(diff[1] / self.max_dist_from_goal, -1, 1)
        obs[2] = np.clip(diff[2] / self.max_dist_from_goal, -1, 1)
        obs[3] = np.clip(self.auv.relative_velocity[0] / 5, -1, 1)  # Forward speed, assuming 5m/s max
        obs[4] = np.clip(self.auv.relative_velocity[1] / 2, -1, 1)  # Side speed, assuming 5m/s max
        obs[5] = np.clip(self.auv.relative_velocity[2] / 2, -1, 1)  # Vertical speed, assuming 5m/s max
        obs[6] = np.clip(self.auv.attitude[0] / self.max_attitude, -1, 1)  # Roll, assuming +-90deg max
        obs[7] = np.clip(self.auv.attitude[1] / self.max_attitude, -1, 1)  # Pitch, assuming +-90deg max
        obs[8] = np.clip(np.sin(self.auv.attitude[2]), -1, 1)  # Yaw, expressed in two polar values to make
        obs[9] = np.clip(np.cos(self.auv.attitude[2]), -1, 1)  # sure observation does not jump between -1 and 1
        obs[10] = np.clip(self.auv.angular_velocity[0] / 1.0, -1, 1)  # Angular Velocities, assuming 1 rad/s
        obs[11] = np.clip(self.auv.angular_velocity[1] / 1.0, -1, 1)
        obs[12] = np.clip(self.auv.angular_velocity[2] / 1.0, -1, 1)

        return obs

    def reward_step(self, action: np.ndarray) -> float:
        """
        Calculate the reward function, make sure to call self.is_done() before to update and check the done conditions

        The factors are defined in the config. Each reward is normalized between 0..1, thus the factor decides its
        importance. Keep in mind the rewards for the done conditions will be sparse.

        Reward 1: Close gto goal location
        Reward 2: Stable attitude
        Reward 3: time step penalty
        Reward 4: action use penalty
        Reward 5: Done - Goal reached
        Reward 6: Done - out of bounds position
        Reward 7: Done - out of bounds attitude
        Reward 8: Done - maximum episode steps

        :param action: array with actions between -1 and 1
        :return: The single reward at this step
        """
        # Reward for being closer to the goal location:
        self.last_reward_arr[0] = ((np.linalg.norm(self.auv.position - self.goal_location)) / self.max_dist_from_goal)**2
        # Reward for stable attitude
        self.last_reward_arr[1] = (np.sum(np.abs(self.auv.attitude[:2]))) / np.pi
        # Negative cum_reward per time step
        self.last_reward_arr[2] = 1
        # Reward for action used (e.g. want to minimize action power usage)
        self.last_reward_arr[3] = np.sum(np.abs(action) * self.action_reward_factors)

        # Add extra reward on checking which condition caused the episode to be done
        self.last_reward_arr[4:] = np.array(self.conditions) * 1

        # Multiply factors defined in config
        self.last_reward_arr = self.last_reward_arr * self.reward_factors

        # Just for analyzing purpose:
        self.cum_reward_arr = self.cum_reward_arr + self.last_reward_arr

        reward = float(np.sum(self.last_reward_arr))

        return reward

    def is_done(self) -> Tuple[bool, list]:
        """
        Condition 0: Check if close to the goal
        Condition 1: Check if out of bounds for position
        Condition 2: Check if attitude (pitch, roll) too high
        Condition 3: Check if maximum time steps reached

        :return: [if simulation is done, extra discrete reward, indexes of conditions that are true]
        """
        # TODO: Collision
        # Calculate distance from goal
        dist_from_goal = np.linalg.norm(self.auv.position - self.goal_location)
        # All conditions in a list
        self.conditions = [
            dist_from_goal < 1.0,  # Condition 0: Check if close to the goal
            dist_from_goal > self.max_dist_from_goal,  # Condition 1: Check if out of bounds for position
            np.any(np.abs(self.auv.attitude[:2]) > self.max_attitude),
            # Condition 2: Check if attitude (pitch, roll) too high
            self.t_steps >= self.max_timesteps  # Condition 3: # Check if maximum time steps reached
        ]

        # If goal reached
        if self.conditions[0]:
            self.goal_reached = True

        # Return also the indexes of which cond is activated
        cond_idx = [i for i, x in enumerate(self.conditions) if x]

        # Check if any condition is true
        done = bool(np.any(self.conditions))  # To satisfy environment checker
        return done, cond_idx

    def render(self, mode="human", real_time=False):
        if real_time:
            plt.pause(self.t_step_size * 0.9)

        if self.episode_data_storage is None:
            self.init_episode_storage()  # The data storage is needed for the plot
        if self.episode_animation is None:
            self.episode_animation = EpisodeAnimation()
            self.ax = self.episode_animation.init_path_animation()
            self.episode_animation.add_episode_text(self.ax, self.episode)
            # Add goal location as tiny sphere, this one is not an obstacle!
            self.episode_animation.add_shapes(self.ax, [shape.Sphere(self.goal_location, 0.5)], 'k')

        self.episode_animation.update_path_animation(positions=self.episode_data_storage.positions,
                                                     attitudes=self.episode_data_storage.attitudes)

        # Possible implementation for rgb_array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array,
        # but not really needed here since 3d.

    def save_full_data_storage(self):
        """
        Call this function to save the full data storage
        """
        self.full_data_storage.save()

    def init_episode_storage(self):
        """
        Small helper function for setting up episode storage when needed
        """
        self.episode_data_storage = EpisodeDataStorage()
        self.episode_data_storage.set_up_episode_storage(path_folder=self.save_path_folder, vehicle=self.auv,
                                                         step_size=self.t_step_size, nu_c_init=self.nu_c,
                                                         shapes=self.obstacles, radar=None, title=self.title,
                                                         episode=self.episode, cum_rewards=self.cum_reward_arr,
                                                         rewards=self.last_reward_arr,
                                                         meta_data_reward=self.meta_data_reward)
