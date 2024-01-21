import copy
from pathlib import Path

import numpy as np

from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import A2C, PPO, DDPG, SAC

from gym_dockauv.config.env_config import TRAIN_CONFIG,TRAIN_CONFIG_remus,TRAIN_CONFIG_remus_Karman
import gym_dockauv.train as train
from gym_dockauv.train import make_gym
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
GYM_ENV = ["ObstaclesCurrentDocking3d_remusStartGoal-v0",] # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
MODELS = [
    SAC,
    # PPO,
]
MODELS_STR = [
    "_SAC",
    # "_PPO",
]
HYPER_PARAMS = [
    SAC_HYPER_PARAMS_TEST,
    PPO_HYPER_PARAMS_TEST,
]
from  PID import PID_Controller
if __name__ == "__main__":
    PID = PID_Controller(0.3, 0.0, 0.00)

    used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
    used_TRAIN_CONFIG["vehicle"] = "remus100"
    start_point = [-10,-8,0]
    goal_point = [-10,8,0]
    used_TRAIN_CONFIG["start_point"] = start_point
    used_TRAIN_CONFIG["goal_point"] = goal_point
    used_TRAIN_CONFIG["bounding_box"] = [26,9,20]
    used_TRAIN_CONFIG["thruster"] = 500
    #计算二范数
    used_TRAIN_CONFIG["max_dist_from_goal"] = np.linalg.norm(np.array(goal_point)-np.array(start_point))
    used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.05*np.linalg.norm(np.array(goal_point)-np.array(start_point))
    used_TRAIN_CONFIG["max_timesteps"] = 1000

    ncount = 1000
    ts = np.zeros(ncount)
    yaw_to_target_list = []
    yaw_control_list = []
    position = []

    if True:
        for K, MODEL in enumerate(MODELS):
            for GYM in GYM_ENV:
                env = make_gym(gym_env=GYM, env_config=used_TRAIN_CONFIG)  # type: BaseDocking3d
                env.reset()
                yaw_to_target = env.yaw_to_target()
                h = 0.02*5
                for i in range(ncount):
                    ts[i] = i * h
                    yaw_control  = PID.control_action(yaw_to_target-0,h)
                    new_obs, rewards, dones, infos = env.step(np.array([yaw_control,0,200]))
                    yaw_to_target = env.yaw_to_target()
                    yaw_to_target_list.append(yaw_to_target)
                    yaw_control_list.append(yaw_control)
                    position.append(infos["position"])

                    if dones:
                        break

                env.reset()
                # 图像输出
                plt.plot(range(len(yaw_to_target_list)), yaw_to_target_list)
                plt.xlabel('x-axis')
                plt.ylabel('y-axis')
                plt.title('yaw error Plot')
                plt.show()

                plt.plot(range(len(yaw_control_list)), yaw_control_list)
                plt.xlabel('x-axis')
                plt.ylabel('y-axis')
                plt.title('yaw control Plot')
                plt.show()
                env.trajectory_in_current(position)

    # ---------- VIDEO GENERATION ----------
    # Example code on how to save a video of on of the saved episode from either prediction or training
    prefix = "/home/ps/dai/overall/togithub/gym_dockauv" \
             "/logs"
             # "/logs/ObstaclesCurrentDocking3d_remus-v0_SAC_10"
    epi_stor = EpisodeDataStorage()
    epi_stor.load(
        file_name=prefix+"/Training Run__EPISODE_1__process_0.pkl")
    epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=80)