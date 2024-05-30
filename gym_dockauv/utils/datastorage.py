import os
import pickle
from copy import deepcopy
import datetime
import numpy as np
import logging
from matplotlib import pyplot as plt

# Used for typehints
from ..objects.auvsim import AUVSim
from ..objects.shape import Shape
from ..objects.sensor import Radar
from typing import List, Type

from .plotutils import EpisodeVisualization

# Set logger
logger = logging.getLogger(__name__)


class FullDataStorage:
    """
    Class to save general simulation data over all the runs/ episodes of simulation:
    - the info dictionary
    - cumulative rewards array

    Pass the created environment by reference and update the Full Data Storage at the end of each episode
    """

    def __init__(self):
        self.file_save_name = None
        self.env = None
        self.storage = None

    def set_up_full_storage(self, env, path_folder: str, title: str = "") -> None:
        r"""
        Set up the storage to save and update incoming data with passing a reference of the env

        file_save_name: will be formatted path_folder\YYYY-MM-DDTHH-MM-SS__{title}__FULL_DATA_STORAGE.pkl

        :param env: The gym environment
        :param path_folder: Path to folder
        :param title: Optional title for addendum
        :return: None
        """
        # Reference the gym environment
        self.env = env

        # Some variables for saving the file
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        if len(path_folder) > 0:
            os.makedirs(path_folder, exist_ok=True)  # Create folder if not exists yet
        self.file_save_name = os.path.join(path_folder, f"{utc_str}__{title}__FULL_DATA_STORAGE.pkl")

        cum_rewards_arr = ArrayList(env.cum_reward_arr)
        rewards_arr = ArrayList(env.last_reward_arr)
        self.storage = {
            "title": title,
            "cum_rewards": cum_rewards_arr,
            "rewards": rewards_arr,
            "meta_data_reward": env.meta_data_reward,
            "n_cont_rewards": env.n_cont_rewards,
            "infos": []
        }

    def update(self) -> None:
        """
        should be called in the end of each episode

        :return: None
        """

        self.storage["cum_rewards"].add_row(self.env.cum_reward_arr)
        self.storage["rewards"].add_row(self.env.last_reward_arr)
        self.storage["infos"].append(self.env.info)

    def save(self) -> str:
        """
        Function to save the pickle file, must be called from the outside, since the gym environment does not know when
        its simulation ends

        :return: path to where file is saved
        """

        self.storage["cum_rewards"] = self.storage["cum_rewards"].get_nparray()
        self.storage["rewards"] = self.storage["rewards"].get_nparray()

        with open(self.file_save_name, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.storage, outp, pickle.HIGHEST_PROTOCOL)

        logger.info(f"Successfully saved FullDataStorage at {self.file_save_name}")

        return self.file_save_name

    def load(self, file_name: str) -> dict:
        """
        Function to load an existing full data storage

        :param file_name: path to file
        :return: the loaded storage
        """
        with open(file_name, 'rb') as inp:
            self.storage = pickle.load(inp)
            return self.storage

    def plot_rewards(self):
        """
        Individual wrapper for plotting rewards
        :return:
        """
        EpisodeVisualization.plot_rewards(cum_rewards=self.storage["cum_rewards"],
                                          rewards=self.storage["rewards"],
                                          episode="all",
                                          title=self.storage["title"],
                                          x_title="episode no.",
                                          meta_data_reward=self.storage["meta_data_reward"],
                                          n_cont_rewards=self.storage["n_cont_rewards"]
                                          )


class ArrayList:
    """
    Custom data type that works as fast as a python list with memory optimization as in numpy

    Background: np.append, np.concatenate, np.vstack are very slow, however, it is necessary for the storage classes
    to grow their array size dynamically and represent the data array so far retrieved for the live animation.
    Already with only 100k function calls it takes a minutes for numpy array to concatenate, while appending list
    like or preallocation takes 0.5 seconds here. More on that here:
    https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    https://stackoverflow.com/questions/46102225/which-one-is-faster-np-vstack-np-append-np-concatenate-or-a-manual-function-ma

    Arrays are saved based on dimension of first vector in nxc format, where c is the length of the initial vector

    :param init_array: any initial array
    """

    def __init__(self, init_array):
        # Initialize data array
        self.dim_col = init_array.shape
        self.capacity = 100
        self.data = np.zeros((self.capacity, *self.dim_col))
        self.size = 1
        self.data[0, :] = init_array

        # Some settings:
        self.array_grow_factor = 4

    def __getitem__(self, index):
        return self.data[:self.size][index]

    def add_row(self, row: np.ndarray) -> None:
        if self.size == self.capacity:
            self.capacity *= self.array_grow_factor
            newdata = np.zeros((self.capacity, *self.dim_col))
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = row
        self.size += 1

    def get_nparray(self) -> np.ndarray:
        return self.data[:self.size]


class EpisodeDataStorage:
    r"""
    Class to save data e.g. vehicle related data during a simulation (e.g. one episode) or env information (e.g.
    reward, observations). Initialize and update after all other initialization and updates

    This data is saved in dictionaries, the keys for retrieval are going to be defined here. This class is highly
    individually written to save gym data and use wrapper around other functions.

    For the definition of the arrays, please refer to the vehicle documentation.

    To add data, if it is an array, use the class ArrayList, which is faster. Make sure to update the setup function,
    update function and the save function accordingly.

    .. note::

        This structure can also be used to load the data and make it more convenient to retrieve specific data again
        by the property functions. However one must keep in mind: during the simulation process, the data type of the
        arrays in the self.storage dictionary are custom defined ArrayList and do not offer all possibilities like
        the numpy arrays.

    Pickle Structure (will be one dict):

    .. code-block:: json

        {
            "vehicle":
            {
                "object": "auvsim.instance (initial one)",
                "states": "auvsim.state nx12 array",
                "states_dot": "auvsim._state_dot nx12 array",
                "u": "auvsim.u nxa array (a number of action)"
            },
            "radar": "Array with radar end points",
            "nu_c": "water current",
            "shapes": "list of shape objects as in shapes.py used here",
            "title": "title of run",
            "episode": "episode number",
            "step_size": "step_size in simulation",
            "cum_rewards": "cumulative reward array",
            "rewards": "reward array per step",
            "meta_data_reward": "List of string for reward description",
            "observation": "Array with observations as in environment"
        }

    """

    def __init__(self):
        self.storage = None
        self.vehicle = None
        self.radar = None
        self.file_save_name = None
        self.env = None

    def set_up_episode_storage(self, path_folder: str, vehicle: AUVSim, step_size: float,
                               nu_c_init: np.ndarray, shapes: List[Shape] = None,
                               radar: Radar = None, title: str = "", episode: int = -1,
                               env=None,
                               index = 0
                               ) -> None:
        r"""
        Set up the storage to save and update incoming data, including passing a reference to the vehicle and
        environment

        file_save_name: will be formatted path_folder\YYYY-MM-DDTHH-MM-SS__{title}__EPISODE_{episode}_DATA_STORAGE.pkl


        :param path_folder: Path to folder
        :param vehicle: Vehicle that is simulated
        :param step_size: Stepsize used in this simulation
        :param nu_c_init: water current information in the simulation (body frame) array 6x1
        :param shapes: Shapes for 3d Animation that were used (static)
        :param radar: Radar sensor if used, save endpoints for post visualization
        :param title: Optional title for addendum
        :param episode: Episode number
        :param env: The gym environment
        :return: None
        """
        # Some variables for saving the file
        utc_str = datetime.datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%S')
        if len(path_folder) > 0:
            os.makedirs(path_folder, exist_ok=True)  # Create folder if not exists yet
        # self.file_save_name = os.path.join(path_folder, f"{utc_str}__{title}__EPISODE_{episode}_DATA_STORAGE.pkl")
        self.file_save_name = os.path.join(path_folder, f"{title}__EPISODE_{episode}__process_{index}.pkl")
        self.index = index
        self.vehicle = vehicle  # Vehicle instance (not a copy, automatically a reference which is updated in reference)
        if shapes is None:
            shapes = []
        self.radar = radar  # Can still be none!
        # Condition statements for saving data in one line
        end_pos_n = ArrayList(radar.end_pos_n) if radar is not None else None
        # Environment safing related stuff
        self.env = env  # This acts as a permanent reference1
        if env is not None:
            cum_rewards_arr = ArrayList(env.cum_reward_arr)
            rewards_arr = ArrayList(env.last_reward_arr)
            meta_data_reward = env.meta_data_reward
            n_cont_rewards = env.n_cont_rewards
            observation = ArrayList(env.observation)
            meta_data_observation = env.meta_data_observation
        else:
            cum_rewards_arr = None
            rewards_arr = None
            meta_data_reward = None
            n_cont_rewards = None
            observation = None
            meta_data_observation = None

        self.storage = {
            "vehicle": {
                "object": vehicle,
                "states": ArrayList(self.vehicle.state),
                "states_dot": ArrayList(self.vehicle._state_dot),
                "u": ArrayList(self.vehicle.u),
            },
            "radar": end_pos_n,
            "nu_c": ArrayList(nu_c_init),
            "shapes": [deepcopy(shape) for shape in shapes],
            "title": title,
            "episode": episode,
            "step_size": step_size,
            "cum_rewards": cum_rewards_arr,
            "rewards": rewards_arr,
            "meta_data_reward": meta_data_reward,
            "n_cont_rewards": n_cont_rewards,
            "observation": observation,
            "meta_data_observation": meta_data_observation,
        }

    def update(self, nu_c: np.ndarray) -> None:
        """
        should be called in the end of each simulation step

        :param nu_c: water current at that time array 6x1
        :param cum_rewards: 1d array with cumulative rewards
        :param rewards: 1d array with rewards
        :return: None
        """
        self.storage["vehicle"]["states"].add_row(self.vehicle.state)
        self.storage["vehicle"]["states_dot"].add_row(self.vehicle._state_dot)
        self.storage["vehicle"]["u"].add_row(self.vehicle.u)
        self.storage["nu_c"].add_row(nu_c)
        if self.env is not None:
            self.storage["cum_rewards"].add_row(self.env.cum_reward_arr)
            self.storage["rewards"].add_row(self.env.last_reward_arr)
            self.storage["observation"].add_row(self.env.observation)
        if self.radar is not None:
            self.storage["radar"].add_row(self.radar.end_pos_n)

    def save(self,cond_idx_done=None,goal_location=None,start_location=None) -> str:
        """
        Function to save the pickle file

        :return: path to where file is saved
        """
        # TODO: think about doing this automatically, but I like explicit
        self.storage["cond_idx_done"] = cond_idx_done
        self.storage["goal_location"] = goal_location
        self.storage["start_location"] = start_location
        self.storage["vehicle"]["states"] = self.storage["vehicle"]["states"].get_nparray()
        self.storage["vehicle"]["states_dot"] = self.storage["vehicle"]["states_dot"].get_nparray()
        self.storage["vehicle"]["u"] = self.storage["vehicle"]["u"].get_nparray()
        if self.env is not None:
            self.storage["cum_rewards"] = self.storage["cum_rewards"].get_nparray()
            self.storage["rewards"] = self.storage["rewards"].get_nparray()
            self.storage["observation"] = self.storage["observation"].get_nparray()
        if self.radar is not None:
            self.storage["radar"] = self.storage["radar"].get_nparray()
        if self.index == 0:
            with open(self.file_save_name, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(self.storage, outp, pickle.HIGHEST_PROTOCOL)

        logger.info(f"Successfully saved EpisodeDataStorage at {self.file_save_name}")

        return self.file_save_name

    def load(self, file_name: str) -> dict:
        """
        Function to load an existing storage

        :param file_name: path to file
        :return: the loaded storage
        """
        with open(file_name, 'rb') as inp:
            self.storage = pickle.load(inp)
            return self.storage

    @property
    def states(self):
        return self.storage["vehicle"]["states"][:]

    @property
    def positions(self) -> np.ndarray:
        r"""
        Returns ALL the position of the AUV in NED coordinates x, y, z

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"][:, 0:3]

    @property
    def attitudes(self) -> np.ndarray:
        r"""
        Returns ALL the attitude (euler angles) of the AUV wrt. to NED coordinates roll, pitch, yaw.

        :return: nx3 array
        """
        return self.storage["vehicle"]["states"][:, 3:6]

    @property
    def step_size(self) -> float:
        return self.storage["step_size"]

    @property
    def u(self) -> np.ndarray:
        return self.storage["vehicle"]["u"][:]

    @property
    def nu_c(self) -> np.ndarray:
        return self.storage["nu_c"][:]

    def plot_episode_animation(self, t_per_step: float = None, title: str = None) -> None:
        """
        Individual wrapper for the animation plot function
        """
        if title is None:
            title = self.storage["title"]
        EpisodeVisualization.plot_episode_animation(
            states=self.states,
            episode=self.storage["episode"],
            shapes=self.storage["shapes"],
            radar_end_pos=self.storage["radar"],
            t_per_step=t_per_step,
            title=title
        )

    def plot_epsiode_states(self):
        """
        Individual wrapper for the static post simulation plot function for states
        :return:
        """
        EpisodeVisualization.plot_episode_states(states=self.states, nu_c=self.nu_c,
                                                 step_size=self.storage["step_size"],
                                                 episode=self.storage["episode"],
                                                 title=self.storage["title"])

    def plot_u(self):
        """
        Individual wrapper for the static post simulation plot function for observations
        :return:
        """
        EpisodeVisualization.plot_u(u=self.u,
                                    step_size=self.storage["step_size"],
                                    episode=self.storage["episode"],
                                    title=self.storage["title"])

    def plot_observation(self):
        """
        Individual wrapper for plotting the observations wih metadata in subplots
        :return:
        """

        EpisodeVisualization.plot_observation(observation=self.storage["observation"],
                                              step_size=self.storage["step_size"],
                                              meta_data_obs=self.storage["meta_data_observation"],
                                              episode=self.storage["episode"],
                                              title=self.storage["title"])

    def plot_rewards(self):
        """
        Individual wrapper for plotting rewards
        :return:
        """
        EpisodeVisualization.plot_rewards(cum_rewards=self.storage["cum_rewards"],
                                          rewards=self.storage["rewards"],
                                          episode=self.storage["episode"],
                                          title=self.storage["title"],
                                          meta_data_reward=self.storage["meta_data_reward"],
                                          n_cont_rewards=self.storage["n_cont_rewards"]
                                          )

    def save_animation_video(self, save_path: str, fps: int) -> None:
        """
        Individual wrapper to save video of loaded storage
        :return:
        """
        x, y, z = self.states[:, 0], self.states[:, 1], self.states[:, 2]

        # # 创建一个 3D 图形
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x, y, z)
        # ax.view_init(elev=94, azim=-89)
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')
        # plt.show()
        #
        # # 创建一个 2D 图形
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(x, y)
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # plt.show()

        EpisodeVisualization.save_animation_video(save_path=save_path,
                                                  fps=fps,
                                                  states=self.states,
                                                  episode=self.storage["episode"],
                                                  # shapes=self.storage["shapes"],
                                                  shapes=None,
                                                  radar_end_pos=self.storage["radar"],
                                                  title=self.storage["title"])
