import copy
from pathlib import Path

import numpy as np

import wandb

from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import A2C, PPO, DDPG, SAC

from gym_dockauv.config.env_config import TRAIN_CONFIG,TRAIN_CONFIG_remus,TRAIN_CONFIG_remus_Karman
import gym_dockauv.train as train
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl
import os

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

# GYM_ENV = ["SimpleDocking3d_remus-v0", "CapsuleDocking3d_remus-v0", "ObstaclesNoCapDocking3d_remus-v0",
#            "ObstaclesDocking3d_remus-v0"]
# MODELS = [
#     SAC,
#     PPO,
# ]
# MODELS_STR = [
#     "_SAC",
#     "_PPO",
# ]
# HYPER_PARAMS = [
#     SAC_HYPER_PARAMS_TEST,
#     PPO_HYPER_PARAMS_TEST,
# ]

# GYM_ENV = ["SimpleCurrentDocking3d-v0",] # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
# GYM_ENV = ["SimpleDocking3d_remus-v0",] # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
GYM_ENV = ["ObstaclesCurrentDocking3d_remusStartGoal-v0",] # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
MODELS = [
    # SAC,
    PPO,
]
MODELS_STR = [
    # "_SAC",
    "_PPO",
]
HYPER_PARAMS = [
    # SAC_HYPER_PARAMS_TEST,
    PPO_HYPER_PARAMS_TEST,
]
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_mode', type=str, default='online')
    args = parser.parse_args()

    # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
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

    wandb.init(project="洋流助力",
               entity="aohuidai",
               # mode=args.wandb_mode,
               mode="online",
               group="卡门我届",
               # name="点到点",
               config=used_TRAIN_CONFIG,

               sync_tensorboard=True,
               # magic=True,
               save_code=True,
               # settings=settings,
               # notes="把熵给改回去了，再跑一个看看",
               )
    if True:
        # if False:
        # ---------- TRAINING ----------
        # Training for multiple models and environment at once
        for K, MODEL in enumerate(MODELS):
            for GYM in GYM_ENV:
                used_TRAIN_CONFIG["title"] = "Training Run"

                log_dir = os.path.join(os.getcwd(), "logs/")
                log_dir = Path(log_dir)
                file_name_prefix = GYM + MODELS_STR[K]
                exst_run_nums = [int(str(folder.name).split(file_name_prefix)[1].split("_")[1]) for folder in
                                 log_dir.iterdir() if
                                 str(folder.name).startswith(file_name_prefix)]
                if len(exst_run_nums) == 0:
                    curr_run = file_name_prefix + "_" + '1'
                else:
                    curr_run = file_name_prefix + "_" + '%i' % (max(exst_run_nums) + 1)
                used_TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs/", curr_run)

                train.train(gym_env=GYM,
                            total_timesteps=10000000,
                            MODEL=MODEL,
                            model_save_path="logs/" + curr_run + "/" + GYM + MODELS_STR[K],
                            tb_log_name=curr_run,

                            agent_hyper_params=HYPER_PARAMS[K],
                            env_config=used_TRAIN_CONFIG,
                            timesteps_per_save=100000,
                            model_load_path=None,
                            # vector_env=16 , )
                            vector_env=None , )
                            # vector_env=1 , )
    else:
        # ---------- VIDEO GENERATION ----------
        # Example code on how to save a video of on of the saved episode from either prediction or training
        prefix = "/home/ps/dai/overall/togithub/gym_dockauv/logs/SimpleDocking3d_remus-v0_SAC_2"
        epi_stor = EpisodeDataStorage()
        epi_stor.load(
            file_name=prefix + "/Training Run__EPISODE_1.pkl")
        epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=20)

    # # Training for one model and one environment
    # train.train(gym_env=GYM_ENV[0],
    #             total_timesteps=50000,
    #             MODEL=SAC,
    #             model_save_path="logs/SAC_docking",
    #             agent_hyper_params=SAC_HYPER_PARAMS_TEST,
    #             env_config=TRAIN_CONFIG,
    #             tb_log_name="SAC",
    #             timesteps_per_save=10000,
    #             model_load_path=None)
    # # Uncomment for plots of previous single run
    # train.post_analysis_directory(directory="./logs",
    #                               show_full=True, show_episode=False)

    # ---------- PREDICTION ----------
    # Prediction for multiple models and environment at once
    # for subdir, dirs, files in os.walk("/home/ps/dai/overall/togithub/gym_dockauv/rew1_final_model"):
    #     for file in sorted(files):
    #         file_split = file.split("_")
    #         ENV = file_split[0]
    #         model_str = file_split[1]
    #         MODEL = PPO if model_str == "PPO" else SAC
    #         # print(MODEL, ENV)
    #         train.predict(gym_env=ENV, model_path=os.path.join(subdir, file), MODEL=MODEL, n_episodes=1000, render=False)

    # Prediction for one model and one environment
    # train.predict(gym_env=GYM_ENV[0], model_path="logs/SAC_docking_50000", MODEL=SAC, n_episodes=3, render=True)
    # # Uncomment for plots of previous single run
    # train.post_analysis_directory(directory="/home/ps/dai/overall/togithub/gym_dockauv/predict_logs")

    # ---------- MANUAL ----------
    # Manual flight in an environment
    # train.manual_control(gym_env=GYM_ENV[3])
    # train.post_analysis_directory(directory="/home/ps/dai/overall/togithub/gym_dockauv/manual_logs")
