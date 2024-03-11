import copy

import gym
import numpy as np

from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import SAC

from gym_dockauv.config.env_config import TRAIN_CONFIG_remus_Karman
from gym_dockauv.train import make_gym
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
GYM_ENV = [
    "ObstaclesCurrentDocking3d_remusStartGoal-v0", ]  # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
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
from gym_dockauv.PID import PID_Controller

if __name__ == "__main__":
    PID = PID_Controller(0.6, 0.00, 0.1)

    # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
    used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
    used_TRAIN_CONFIG["vehicle"] = "remus100"
    start_point = [-0, -10, 0]
    goal_point = [-0, 10, 0]
    used_TRAIN_CONFIG["start_point"] = start_point
    used_TRAIN_CONFIG["goal_point"] = goal_point
    used_TRAIN_CONFIG["bounding_box"] = [26, 18, 20]
    used_TRAIN_CONFIG["thruster"] = 1000
    # 计算二范数
    used_TRAIN_CONFIG["max_dist_from_goal"] = np.linalg.norm(np.array(goal_point) - np.array(start_point))
    used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.05 * np.linalg.norm(np.array(goal_point) - np.array(start_point))
    used_TRAIN_CONFIG["max_timesteps"] = 1000

    used_TRAIN_CONFIG["title"] = "Training Run"

    ncount = 1000
    ts = np.zeros(ncount)
    yaw_to_target_list = []
    yaw_control_list = []
    all_positions = []
    position = []

    if True:
        for K, MODEL in enumerate(MODELS):
            for GYM in GYM_ENV:
                """
                PID position
                """
                env = make_gym(gym_env=GYM, env_config=used_TRAIN_CONFIG)  # type: BaseDocking3d
                env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.ClipAction(env)
                env = gym.wrappers.NormalizeObservation(env)
                env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
                env = gym.wrappers.NormalizeReward(env, gamma=0.99)
                env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

                env.reset()
                yaw_to_target = env.yaw_to_target()
                h = 0.02 * 5
                for i in range(ncount):
                    ts[i] = i * h
                    yaw_control = PID.control_action(yaw_to_target - 0, h)
                    new_obs, rewards, dones, infos = env.step(np.array([yaw_control, 0, 600]))
                    yaw_to_target = env.yaw_to_target()
                    yaw_to_target_list.append(yaw_to_target)
                    yaw_control_list.append(yaw_control)
                    position.append(infos["position"])

                    if dones:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        print("charts/episodic_return", infos["episode"]["r"])
                        print("charts/episodic_length", infos["episode"]["l"])

                        print("thruster/thruster", infos["thruster"])
                        print("thruster/thruster_square", infos["thruster_square"])
                        print("thruster/thruster_mean", infos["thruster"] / infos["episode"]["l"],
                              )
                        print("charts/total_distance_moved", infos["total_distance_moved"])
                        print("charts/velocity=total_distance_moved/t",
                              infos["total_distance_moved"] / (0.1 * infos["episode"]["l"]),
                              )
                        distance_moved = [np.linalg.norm(position[i + 1] - position[i]) for i in
                                          range(len(position) - 1)]
                        print("*****************")
                        print("steps", len(position))
                        print("total_distance_moved", sum(distance_moved))
                        print("total_distance_moved/t", sum(distance_moved) / (0.1 * len(position)))
                        PID_steps = len(position)
                        PID_total_distance_moved = sum(distance_moved)

                        break

                env.reset()
                # # 图像输出
                # plt.plot(range(len(yaw_to_target_list)), yaw_to_target_list)
                # plt.xlabel('x-axis')
                # plt.ylabel('y-axis')
                # plt.title('yaw error Plot')
                # plt.show()
                #
                # plt.plot(range(len(yaw_control_list)), yaw_control_list)
                # plt.xlabel('x-axis')
                # plt.ylabel('y-axis')
                # plt.title('yaw control Plot')
                # plt.show()
                all_positions.append(np.array(position))

                """
                RL plot
                """
                prefix = "./ObstaclesCurrentDocking3d_remusStartGoal-v0ppo_continuous_action_20"
                # "/logs/ObstaclesCurrentDocking3d_remusStartGoal-v0ppo_continuous_action_50"
                # "/logs"
                epi_stor = EpisodeDataStorage()
                epi_stor.load(
                    file_name=prefix + f"/Training Run__EPISODE_499__process_0.pkl")
                    # file_name=prefix + f"/Training Run__EPISODE_300__process_0.pkl")
                position = epi_stor.positions
                all_positions.append(position)
                distance_moved = [np.linalg.norm(position[i + 1] - position[i]) for i in range(len(position) - 1)]
                print("*****************")
                print("steps", len(position))
                print("total_distance_moved", sum(distance_moved))
                print("total_distance_moved/t", sum(distance_moved) / (0.1 * len(position)))
                RL_steps = len(position)
                RL_total_distance_moved = sum(distance_moved)

                # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
                # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
                # used_TRAIN_CONFIG["vehicle"] = "remus100"
                # start_point = [-0, -100, 0]
                # goal_point = [-0, 100, 0]
                # used_TRAIN_CONFIG["start_point"] = start_point
                # used_TRAIN_CONFIG["goal_point"] = goal_point
                # used_TRAIN_CONFIG["bounding_box"] = [260, 90 * 2, 200]
                # used_TRAIN_CONFIG["thruster"] = 500

                env = make_gym(gym_env=GYM_ENV[0], env_config=used_TRAIN_CONFIG)  # type: BaseDocking3d
                env.trajectory_in_current(all_positions, prefix=prefix + "trajectory_PID_RL",
                                          args={"multi": True,"start_point":[0,-100,0],"goal_point":[0,100,0],
                                                "label": [r"PID,n=600",r"OCDRP w/ R_{thruster}",r"OCDRP w/o R_{thruster},n=400"],
                                                "annotated": [f"steps:{PID_steps}\n path length:{round(10*PID_total_distance_moved,1)}",
                                                    f"steps:{RL_steps} \n path length:{round(10* RL_total_distance_moved,1)}"],
                                                })

                # env.trajectory_in_current(position)

    # ---------- VIDEO GENERATION ----------
    # Example code on how to save a video of on of the saved episode from either prediction or training
    prefix = "/home/ps/dai/overall/togithub/gym_dockauv" \
             "/logs"
    # "/logs/ObstaclesCurrentDocking3d_remus-v0_SAC_10"
    epi_stor = EpisodeDataStorage()
    epi_stor.load(
        file_name=prefix + "/Training Run__EPISODE_1__process_0.pkl")
    epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=80)
