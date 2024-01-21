import copy
from pathlib import Path

from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
from stable_baselines3 import A2C, PPO, DDPG, SAC
from gym_dockauv.config.env_config import PREDICT_CONFIG, MANUAL_CONFIG, TRAIN_CONFIG, REGISTRATION_DICT,TRAIN_CONFIG_remus_Karman
import gym
from gym_dockauv.config.env_config import TRAIN_CONFIG
# import gym_dockauv.train as train
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl
import os

mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
def make_gym(gym_env: str, env_config: dict):
    """
    Wrapper to create and return gym and return error if key is wrong

    :param gym_env: Registration string of gym from docking3d env
    :param env_config: Config for environment
    :return:
    """
    if gym_env in REGISTRATION_DICT:
        env = gym.make(gym_env, env_config=env_config)
        return env
    else:
        raise KeyError(f"Not valid gym environment registration string,"
                       f" available options are {REGISTRATION_DICT.keys()}")


GYM_ENV = ["ObstaclesCurrentDocking3d_remus-v0",] # "SimpleDocking3d-v0",  "CapsuleDocking3d-v0", "ObstaclesNoCapDocking3d-v0", "ObstaclesDocking3d-v0"]
MODELS = [
    SAC,
    PPO,
]
MODELS_STR = [
    "_SAC",
    "_PPO",
]
HYPER_PARAMS = [
    SAC_HYPER_PARAMS_TEST,
    PPO_HYPER_PARAMS_TEST,
]

if __name__ == "__main__":
    TRAIN_CONFIG["vehicle"] = "remus100"
    # if True:
    if False:
        # ---------- TRAINING ----------
        # Training for multiple models and environment at once
        for GYM in GYM_ENV:
            for K, MODEL in enumerate(MODELS):
                TRAIN_CONFIG["title"] = "Training Run"

                log_dir = os.path.join(os.getcwd(), "logs/")
                log_dir = Path(log_dir)
                file_name_prefix = GYM + MODELS_STR[K]
                exst_run_nums = [int(str(folder.name).split(file_name_prefix)[1].split("_")[1]) for folder in log_dir.iterdir() if
                                 str(folder.name).startswith(file_name_prefix)]
                if len(exst_run_nums) == 0:
                    curr_run = file_name_prefix + "_" + '1'
                else:
                    curr_run = file_name_prefix + "_" + '%i' % (max(exst_run_nums) + 1)
                TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs/",curr_run)

                train.train(gym_env=GYM,
                            total_timesteps=1000000,
                            MODEL=MODEL,
                            model_save_path="logs/"+curr_run+GYM+MODELS_STR[K],
                            tb_log_name=curr_run,

                            agent_hyper_params=HYPER_PARAMS[K],
                            env_config=TRAIN_CONFIG,
                            timesteps_per_save=100000,
                            model_load_path=None)
    else:
        # ---------- VIDEO GENERATION ----------
        # Example code on how to save a video of on of the saved episode from either prediction or training
        for i in range(1,20):
            for j in range(0,1):
                prefix = "/home/ps/dai/overall/togithub/gym_dockauv" \
                         "/logs/ObstaclesCurrentDocking3d_remusStartGoal-v0_SAC_21"
                         # "/logs"
                epi_stor = EpisodeDataStorage()
                epi_stor.load(
                    file_name=prefix+f"/Training Run__EPISODE_{i}__process_{j}.pkl")

                # used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
                used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
                used_TRAIN_CONFIG["vehicle"] = "remus100"
                start_point = [-100, -80, 0]
                goal_point = [-100, 80, 0]
                used_TRAIN_CONFIG["start_point"] = start_point
                used_TRAIN_CONFIG["goal_point"] = goal_point
                used_TRAIN_CONFIG["bounding_box"] = [260, 90, 200]
                used_TRAIN_CONFIG["thruster"] = 500

                env = make_gym(gym_env=GYM_ENV[0], env_config=used_TRAIN_CONFIG)  # type: BaseDocking3d
                env.trajectory_in_current(epi_stor.positions,prefix=prefix+f"/fig_episode_{i}_process{j}")

        epi_stor.save_animation_video(save_path="goal_constr_fail.mp4", fps=80)

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


