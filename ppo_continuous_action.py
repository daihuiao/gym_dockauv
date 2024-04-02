# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy


from huggingface_hub.utils import tqdm

from gym_dockauv.utils.datastorage import EpisodeDataStorage, FullDataStorage
from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_DEFAULT
from gym_dockauv.config.env_config import PREDICT_CONFIG, MANUAL_CONFIG, TRAIN_CONFIG, REGISTRATION_DICT
from gym_dockauv.envs.docking3d import BaseDocking3d

import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gym
# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gym_dockauv.evaluation import evaluate

import numpy as np

import wandb

from gym_dockauv.config.DRL_hyperparams import PPO_HYPER_PARAMS_TEST, SAC_HYPER_PARAMS_TEST
# from stable_baselines3 import A2C, PPO, DDPG, SAC

from gym_dockauv.config.env_config import TRAIN_CONFIG, TRAIN_CONFIG_remus, TRAIN_CONFIG_remus_Karman
import gym_dockauv.train as train
from gym_dockauv.utils.datastorage import EpisodeDataStorage
import matplotlib as mpl
import os
from stable_baselines3.common.vec_env import SubprocVecEnv

os.environ["WANDB_API_KEY"] = "b4fdd4e5e894cba0eda9610de6f9f04b87a86453"


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "0301_clip"
    """the wandb's project name"""
    wandb_entity: str = "aohuidai"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "ObstaclesCurrentDocking3d_remusStartGoal-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 12
    num_envs_eval: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    current_on: bool = True
    tau: float = 0.5

    w_velocity: float = 0.1
    thruster_penalty: float = 1.0
    thruster: float = 400
    thruster_min: float = 0.

def make_env(env_id, index, capture_video, run_name, gamma, env_config):
    def thunk():
        env_config_ = copy.deepcopy(env_config)
        env_config_["index"] = index

        if capture_video and index == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, env_config=env_config_)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG)
used_TRAIN_CONFIG = copy.deepcopy(TRAIN_CONFIG_remus_Karman)
used_TRAIN_CONFIG["vehicle"] = "remus100"
start_point = [-0, -10, 0]
goal_point = [-0, 10, 0]
used_TRAIN_CONFIG["start_point"] = start_point
used_TRAIN_CONFIG["goal_point"] = goal_point
used_TRAIN_CONFIG["bounding_box"] = [26, 18, 20]
# 计算二范数
used_TRAIN_CONFIG["max_dist_from_goal"] = np.linalg.norm(np.array(goal_point) - np.array(start_point))
used_TRAIN_CONFIG["dist_goal_reached_tol"] = 0.05 * np.linalg.norm(np.array(goal_point) - np.array(start_point))
used_TRAIN_CONFIG["max_timesteps"] = 999

used_TRAIN_CONFIG["title"] = "Training Run"

log_dir = os.path.join(os.getcwd(), "logs/")
log_dir = Path(log_dir)
file_name_prefix = Args.env_id + "ppo_continuous_action"
exst_run_nums = [int(str(folder.name).split(file_name_prefix)[1].split("_")[1]) for folder in
                 log_dir.iterdir() if
                 str(folder.name).startswith(file_name_prefix)]
if len(exst_run_nums) == 0:
    curr_run = file_name_prefix + "_" + '1'
else:
    curr_run = file_name_prefix + "_" + '%i' % (max(exst_run_nums) + 1)

if __name__ == "__main__":
    args = tyro.cli(Args)

    used_TRAIN_CONFIG["current_on"] = args.current_on
    used_TRAIN_CONFIG["reward_factors"]["w_velocity"] = args.w_velocity
    used_TRAIN_CONFIG["reward_factors"]["thruster_penalty"] = args.thruster_penalty
    used_TRAIN_CONFIG["thruster"] = args.thruster
    used_TRAIN_CONFIG["thruster_min"] = args.thruster_min

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.env_id.split()}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # run_name = f"tau:{args.tau}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"tau:{args.tau}_currentOn:{args.current_on}_w_velocity:{args.w_velocity}" \
               f"_thruster:{args.thruster_penalty}_{args.seed}__{int(time.time())}"
    # used_TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs/", curr_run)
    used_TRAIN_CONFIG["save_path_folder"] = os.path.join(os.getcwd(), "logs/", run_name)

    print("current on:",args.current_on)
    if args.track:
        import wandb

        my_config = vars(args)
        my_config.update(used_TRAIN_CONFIG)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group="cleanrl" + "_tau:" + str(args.tau)+"clip:"+str(args.clip_coef),
            mode="online",
            sync_tensorboard=True,
            config=my_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,used_TRAIN_CONFIG) for i in range(args.num_envs)]
    # )
    envs = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, used_TRAIN_CONFIG) for i in
            range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    # envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs_eval = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, used_TRAIN_CONFIG) for i in
            range(1,args.num_envs_eval+1)]
    envs_eval = SubprocVecEnv(envs_eval)
    envs_eval.single_action_space = envs_eval.action_space
    envs_eval.single_observation_space = envs_eval.observation_space

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs, _ = envs.reset(seed=args.seed)
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    low, high = torch.tensor(envs.action_space.low).to(device), torch.tensor(envs.action_space.high).to(device)
    tqdm_bar = tqdm(total=1e6, desc="Training", unit=" steps", ncols=100)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            tqdm_bar.update(args.num_envs)

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            action = low + (0.5 * (action + 1.0) * (high - low))
            next_obs, reward, terminations, infos = envs.step(action.cpu().numpy())
            # for i in range(args.num_envs):
            #     if terminations[i]:
            #         next_obs[i] = envs.reset(i)
            truncations = False
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for i in range(args.num_envs):
                if "episode" in infos[i]:
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", infos[i]["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", infos[i]["episode"]["l"], global_step)

                    writer.add_scalar("thruster/thruster", infos[i]["thruster"], global_step)
                    writer.add_scalar("thruster/thruster_square", infos[i]["thruster_square"], global_step)
                    writer.add_scalar("thruster/thruster_mean", infos[i]["thruster"] / infos[i]["episode"]["l"],
                                      global_step)
                    writer.add_scalar("charts/total_distance_moved",infos[i]["total_distance_moved"],global_step)
                    writer.add_scalar("charts/velocity=total_distance_moved/t",infos[i]["total_distance_moved"]/ (0.1*infos[i]["episode"]["l"]),global_step)
            # indices = np.where(infos[0]["conditions_true"])
            indices = infos[0]["conditions_true"]
            if len(indices) > 0:
                try:
                    wandb.log({"logs/done_condition(0:goal,1:border,3:step,4：collision)": indices[0]})
                except:
                    pass

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = asymmetric_l2_loss(v_clipped - b_returns[mb_inds], args.tau)
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        #这里该评估一下了
        episode_return, episode_length,evaluation_time = evaluate(envs_eval,agent,device=device,args=args,
                                          low=low,high=high,num_envs=args.num_envs_eval,writer=writer,global_step=global_step)
        # writer.add_scalar("evaluation/eval_episodic_return", episode_return, global_step)
        # writer.add_scalar("evaluation/eval_episodic_length", episode_length, global_step)
        # writer.add_scalar("evaluation/eval_time", evaluation_time, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
