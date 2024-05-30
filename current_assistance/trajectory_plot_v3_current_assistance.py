import copy
import pickle

import gym
import numpy as np
from matplotlib import pyplot as plt

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
    PID = PID_Controller(0.3, 0.00, 0.01)

    # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
    used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
    used_TRAIN_CONFIG["vehicle"] = "remus100"
    start_point = [-0, -5, 0]
    goal_point = [-0, 5, 0]
    used_TRAIN_CONFIG["start_point"] = start_point
    used_TRAIN_CONFIG["goal_point"] = goal_point
    used_TRAIN_CONFIG["bounding_box"] = [26, 18, 20]
    used_TRAIN_CONFIG["thruster"] = 300
    # 计算二范数
    used_TRAIN_CONFIG["max_dist_from_goal"] = np.linalg.norm(np.array(goal_point) - np.array(start_point))
    used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.08 * np.linalg.norm(np.array(goal_point) - np.array(start_point))
    used_TRAIN_CONFIG["max_timesteps"] = 1999

    used_TRAIN_CONFIG["title"] = "Training Run"

    ncount = 2000
    ts = np.zeros(ncount)


    if True:
        for K, MODEL in enumerate(MODELS):
            for GYM in GYM_ENV:
                """
                PID position
                """
                yaw_to_target_list = []
                yaw_control_list = []
                all_positions = []
                position = []
                state12s =[]
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
                    thruster_speed=300

                    new_obs, rewards, dones, infos = env.step(np.array([yaw_control, 0, thruster_speed]))
                    yaw_to_target = env.yaw_to_target()
                    yaw_to_target_list.append(yaw_to_target)
                    yaw_control_list.append(yaw_control)
                    position.append(infos["position"])
                    state12s.append(infos["state12"])


                    if dones:

                        haha = np.array(state12s)
                        thruster_1 = np.ones(len(position)) * thruster_speed
                        with open("PID2000.pkl", "wb") as f:
                            pickle.dump(haha, f)
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        distance_moved_1 = [np.linalg.norm(position[i + 1] - position[i]) for i in
                                          range(len(position) - 1)]
                        # distacne_moved补齐
                        distance_moved_1.append(0)
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
                        print("PID")
                        print("steps", len(position))
                        print("total_distance_moved", sum(distance_moved))
                        velocity_1_total =sum(distance_moved) / (0.1 * len(position))
                        print("total_distance_moved/t", sum(distance_moved) / (0.1 * len(position)))
                        PID_steps = len(position)
                        PID_total_distance_moved = sum(distance_moved)

                        break

                env.reset()

# todo attention
                prefix = "./logs"
                epi_stor = EpisodeDataStorage()
                epi_stor.load(
                    file_name=prefix + f"/Training Run__EPISODE_1__process_0.pkl")
                epi_stor.save_animation_video(save_path="animation/PID.mp4", fps=80)

                all_positions.append(np.array(position))
                position_0 = copy.deepcopy(position)


                """
                 speed plot
                """

                prefix = "./logs/ObstaclesCurrentDocking3d_remusStartGoal-v0_SAC_24"

                epi_stor = EpisodeDataStorage()
                epi_stor.load(
                    file_name=prefix + f"/Training Run__EPISODE_150__process_0.pkl")

                thruster = epi_stor.u[:, 2]
                thruster_2 = epi_stor.u[:, 2]
                # todo dai: 可能有问题，这个是自身坐标系下的速度大小，洋流?
                velocity = epi_stor.states[:, 6:9]
                velocity_norm_2 = np.linalg.norm(velocity, axis=1)

                distance_moved_2 = [np.linalg.norm(epi_stor.positions[i + 1] - epi_stor.positions[i]) for i in
                                    range(len(epi_stor.positions) - 1)]
                distance_moved_2.append(distance_moved_2[-1])

                position = epi_stor.positions
                pos_len = len(position)
                # position = position[int(0.6*pos_len):int(0.8*pos_len)]
                distance_moved = [np.linalg.norm(position[i + 1] - position[i]) for i in range(len(position) - 1)]
                print("thruster adaptive")
                print("steps", len(position))
                print("total_distance_moved", sum(distance_moved))
                velocity_2_total = sum(distance_moved) / (0.1 * len(position))
                print("total_distance_moved/t", velocity_2_total)
                RL_steps = len(position)
                RL_total_distance_moved = sum(distance_moved)
                all_positions.append(np.array(position))
#todo attention
                epi_stor.save_animation_video(save_path="animation/goal_constr_fail.mp4", fps=80)

                env = make_gym(gym_env=GYM_ENV[0], env_config=used_TRAIN_CONFIG)  # type: BaseDocking3d
                # if False:
                if True:
                    env.trajectory_in_current_1(all_positions
                                               , prefix=prefix + f"/haha",
                                               args={"fig.16_1": True,
                                                     "multi": True, "start_point": [0, -50, 0],
                                                     "goal_point": [0, 50, 0],
                                                     "label": [r"PID,n=300", r"RL, $n$=300"],
                                                     "annotated": [
                                                         f"steps:{PID_steps}\n path length:{round(10 * PID_total_distance_moved, 1)} \n velocity{round(velocity_1_total, 1)}",
                                                         f"steps:{RL_steps} \n path length:{round(10 * RL_total_distance_moved, 1)}\n velocity{round(velocity_2_total, 1)}"],
                                                     },
                                               position1=position,
                                               args1={"cmap": True, "cmap_value": distance_moved_2,
                                                      "cmap_value_1": distance_moved_1, }
                                               # args1={"cmap": True, "cmap_value": thruster_2,
                                               #        "cmap_value_1": thruster_1, }
                                               )
                if True:  # thruster plot
                    # plt.close()
                    # plt.figure(figsize=(4, 4))
                    plt.plot(range(len(distance_moved_2)), distance_moved_2, label="thruster speed \n adaptative", c="orange")
                    plt.xlabel("step", fontsize='x-large')
                    plt.ylabel("thruster speed n", fontsize='x-large')
                    # plt.savefig("thruster speed in an episode.png")
                    plt.legend(loc="upper center", bbox_to_anchor=(0.48, 0.9),
                               fontsize='x-large')  # lower center', 'upper center
                    # plt.show()
                    plt.savefig(f"./thurster_speed_1_719.pdf", bbox_inches='tight')
                    haha = True
                # env.trajectory_in_current(position)


                # env.trajectory_in_current(position)

    # ---------- VIDEO GENERATION ----------
    # Example code on how to save a video of on of the saved episode from either prediction or training
    # prefix = "/home/ps/dai/overall/togithub/gym_dockauv" \
    #          "/logs"
    # # "/logs/ObstaclesCurrentDocking3d_remus-v0_SAC_10"
    # epi_stor = EpisodeDataStorage()
    # epi_stor.load(
    #     file_name=prefix + "/Training Run__EPISODE_1__process_0.pkl")
    epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=80)
