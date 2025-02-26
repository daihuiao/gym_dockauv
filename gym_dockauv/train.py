import copy
import os
import logging

import numpy as np

import gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common import base_class
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

from .utils.datastorage import EpisodeDataStorage, FullDataStorage
from .config.DRL_hyperparams import PPO_HYPER_PARAMS_DEFAULT
from .config.env_config import PREDICT_CONFIG, MANUAL_CONFIG, TRAIN_CONFIG, REGISTRATION_DICT
from .envs.docking3d import BaseDocking3d

# Set logger
logger = logging.getLogger(__name__)


def train(gym_env: str,
          total_timesteps: int,
          MODEL: base_class = PPO,
          model_save_path: str = "logs/PPO_docking",
          agent_hyper_params: dict = PPO_HYPER_PARAMS_DEFAULT,
          env_config: dict = TRAIN_CONFIG,
          tb_log_name: str = "PPO",
          timesteps_per_save: int = None,
          model_load_path: str = None,
          vector_env: int = None) -> None:
    f"""
    Function to train and save model, own wrapper
    
    Model name that will be saved is "[model_save_path]_[elapsed_timesteps]", when timesteps_per_save is given model 
    is captured and saved in between 
    
    .. note:: Interval of saving and number of total runtime might be inaccurate, if the StableBaseLine agent n_steps 
        is not accordingly updated, for example total runtime is 3000 steps, however, update per n_steps of the agent is 
        by default for PPO at 2048, thus the agents only checks if its own simulation time steps is bigger than 3000 
        after every multiple of 2048 

    :param MODEL: DRL algorithm model to use
    :param gym_env: Registration string of gym from docking3d
    :param total_timesteps: total timesteps for this training run
    :param model_save_path: path where to save the model
    :param agent_hyper_params: agent hyper parameter, default is always loaded
    :param env_config: environment configuration
    :param tb_log_name: log file name of this run for tensor board
    :param timesteps_per_save: simulation timesteps before saving the model in that interval
    :param model_load_path: path of existing model, use to continue training with that model
    :return: None
    """
    # Create environment
    if vector_env is not None:
        def make_env_(index):
            def _init():
                env_config_ = copy.deepcopy(env_config)
                env_config_["index"] = index
                env = make_gym(gym_env=gym_env, env_config=env_config_)  # type: BaseDocking3d
                # env = Monitor(env, './sb3logs/')  # 设置日志文件夹
                env = Monitor(env)  # 设置日志文件夹
                return env
            return _init

        envs = [make_env_(_) for _ in range(vector_env)]
        # env = DummyVecEnv(envs)
        env = SubprocVecEnv(envs)
    else:
        env = make_gym(gym_env=gym_env, env_config=env_config)  # type: BaseDocking3d

    # Init variables
    elapsed_timesteps = 0
    sim_timesteps = timesteps_per_save if timesteps_per_save else total_timesteps

    # # For DDPG algorithm
    # n_actions = env.action_space.shape[0]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Instantiate the agent
    if model_load_path is None:
        model = MODEL(policy='MlpPolicy', env=env, **agent_hyper_params)
    else:
        # Note that this does not load a replay buffer
        model = MODEL.load(model_load_path, env=env)

    while elapsed_timesteps < total_timesteps:
        # Train the agent
        model.learn(total_timesteps=sim_timesteps, reset_num_timesteps=False, tb_log_name=tb_log_name)
        # Taking the actual elapsed timesteps here, so the total simulation time at least will not be biased
        elapsed_timesteps = model.num_timesteps
        # Save the agent
        tmp_model_save_path = f"{model_save_path}_{elapsed_timesteps}"
        # This DOES NOT save the replay/rollout buffer, that is why we continue using the same model instead of
        # reloading anything in the while loop
        model.save(tmp_model_save_path)
        logger.info(f'Successfully saved model: {os.path.join(os.path.join(os.getcwd(), tmp_model_save_path))}')

    env.save_full_data_storage()
    return None


# TODO
def predict(gym_env: str, model_path: str, MODEL: base_class = PPO, n_episodes: int = 5, render: bool = True):
    """
    Function to visualize and evaluate the actual model on the environment

    :param gym_env: Registration string of gym from docking3d
    :param model_path: full path of trained agent
    :param MODEL: stable baseline model
    :param n_episodes: number of episodes to run
    :param render: boolean for render
    :return:
    """
    env = make_gym(gym_env=gym_env, env_config=PREDICT_CONFIG)  # type: BaseDocking3d
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one

    n_actions = env.action_space.shape[0]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = MODEL.load(model_path, env=env)

    # Enjoy trained agent
    obs = env.reset(seed=2)
    for i in range(n_episodes):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if render:
                env.render(rotate_cam=True)
            if done:
                break
        env.reset()
    env.save_full_data_storage()


def post_analysis_directory(directory: str = "/home/erikx3/PycharmProjects/gym_dockauv/logs", show_full: bool = True,
                            show_episode: bool = True):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # Capture full data pkl file
        full_path = os.path.join(directory, filename)
        if filename.endswith("FULL_DATA_STORAGE.pkl") and show_full:
            full_stor = FullDataStorage()
            full_stor.load(full_path)
            full_stor.plot_rewards()
            plt.show()
        # Episode Data Storage:
        elif filename.endswith(".pkl") and show_episode:
            epi_stor = EpisodeDataStorage()
            epi_stor.load(full_path)
            epi_stor.plot_epsiode_states()
            epi_stor.plot_u()
            epi_stor.plot_observation()
            epi_stor.plot_rewards()
            plt.show()
            # epi_stor.plot_episode_animation(t_per_step=None, title="Test Post Flight Visualization")


def manual_control(gym_env: str):
    """
    Function with pygame workaround to manually fly and debug the vehicle

    Great for debugging purposes, since post analysis can be called on the log that is created

    :param gym_env: Registration string of gym from docking3d env
    """
    import pygame

    # Settings:
    WINDOW_X = 600
    WINDOW_Y = 400
    # Init environment
    env = make_gym(gym_env=gym_env, env_config=MANUAL_CONFIG)  # type: BaseDocking3d
    env.reset()
    done = False
    # Init pygame
    pygame.init()
    window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))
    run = True
    # Init pygame text I want to use
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    text_title = my_font.render('Click on this window to control vehicle', False, (255, 255, 255))
    text_note = my_font.render('Note: Not real time! Press keys below.', False, (255, 255, 255))
    text_instructions1 = [
        my_font.render('Input 1 / Linear x:', False, (255, 255, 255)),
        my_font.render('Input 2 / Linear y:', False, (255, 255, 255)),
        my_font.render('Input 3 / Linear z:', False, (255, 255, 255)),
        my_font.render('Input 4 / Angular x:', False, (255, 255, 255)),
        my_font.render('Input 5 / Angular y:', False, (255, 255, 255)),
        my_font.render('Input 6 / Angular z:', False, (255, 255, 255))]
    text_instructions2 = [my_font.render('w', False, (255, 255, 255)),
                          my_font.render('a', False, (255, 255, 255)),
                          my_font.render('f', False, (255, 255, 255)),
                          my_font.render('u', False, (255, 255, 255)),
                          my_font.render('h', False, (255, 255, 255)),
                          my_font.render('o', False, (255, 255, 255))]
    text_instructions3 = [my_font.render('s', False, (255, 255, 255)),
                          my_font.render('d', False, (255, 255, 255)),
                          my_font.render('r', False, (255, 255, 255)),
                          my_font.render('j', False, (255, 255, 255)),
                          my_font.render('k', False, (255, 255, 255)),
                          my_font.render('l', False, (255, 255, 255))]
    # Init valid action
    action = np.zeros(6)
    valid_input_no = env.auv.u_bound.shape[0]
    while run:
        # --------- Pygame ----------
        # Text and shapes on pygame window
        window.fill((0, 0, 0))  # Make black background again
        window.blit(text_title, (0, 0))
        window.blit(text_note, (0, 30))
        pos_y = 80
        count = 0
        for text1, text2, text3 in zip(text_instructions1, text_instructions2, text_instructions3):
            window.blit(text1, (0, pos_y))
            window.blit(text2, (250, pos_y))
            window.blit(text3, (WINDOW_X - 50, pos_y))
            # Here we draw the circle based on which keyboard are pressed
            circle_x = WINDOW_X - 100 - (WINDOW_X - 100 - 300) / 2 * (action[count] + 1)
            pygame.draw.circle(window, 'green', (circle_x, pos_y + 10), 5)
            pos_y += 45
            count += 1
        # Draw a green line
        line_x = (250 + WINDOW_X - 50) / 2
        pygame.draw.line(window, 'green', (line_x, 80), (line_x, 80 + 5 * 45 + 30))
        # Update pygame window
        pygame.display.update()
        pygame.display.flip()

        # --------- Derive action from keyboard input ---------
        action = np.zeros(6)
        keys = pygame.key.get_pressed()
        action[0] = (keys[pygame.K_w] - keys[pygame.K_s]) * 1
        action[1] = (keys[pygame.K_a] - keys[pygame.K_d]) * 1
        action[2] = (keys[pygame.K_f] - keys[pygame.K_r]) * 1
        action[3] = (keys[pygame.K_u] - keys[pygame.K_j]) * 1
        action[4] = (keys[pygame.K_h] - keys[pygame.K_k]) * 1
        action[5] = (keys[pygame.K_o] - keys[pygame.K_l]) * 1

        # Get valid action, as number of inputs might be smaller than 6 for other vehicles
        valid_action = action[:valid_input_no]

        # Need this part below to make everything work, but also quitting
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.type == pygame.QUIT:
                    run = False

        # --------- Environment ---------
        if not done:
            # Env related stuff
            obs, rewards, done, info = env.step(valid_action)
            env.render()
        # This last call ensures, that we save a log for "episode one"
        else:
            env.reset()
            done = False
    # Call in case of quit
    env.reset()


def make_gym(gym_env: str, env_config: dict) -> BaseDocking3d:
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
