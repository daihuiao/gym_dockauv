import time

import numpy as np
import torch


@torch.no_grad()
def evaluate(envs_eval,agent,device,args,low,high,num_envs,writer,global_step):
# with torch.no_grad():

# ALGO Logic: Storage setup
#     obs = torch.zeros((args.num_steps, args.num_envs) + envs_eval.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    t = time.time()
    unfinished = np.ones(num_envs).astype(bool)
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_return_env_computed = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, args.num_steps)

    # next_done = torch.zeros(10).to(device)
    next_obs = envs_eval.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    haha = False
    for step in range(0, args.num_steps):
        # obs[step] = next_obs
        # dones[step] = next_done

        # action, logprob, _, value = agent.get_action_and_value(next_obs)
        #todo dai：不要随机性，只拿均值，如果初始状态没有随机性，这里有点冗余
        action = agent.actor_mean(next_obs)
        # values[step] = value.flatten()
        # actions[step] = action
        # logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        action = low + (0.5 * (action + 1.0) * (high - low))


        next_obs, reward, next_done_, infos = envs_eval.step(action.cpu().numpy())
        # rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done_).to(device)
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
#todo dai :现在只能测试单个的环境！！！
        for i in range(num_envs):
            if "episode" in infos[i]:
                haha = True
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("evaluation/episodic_return", infos[i]["episode"]["r"], global_step)

                writer.add_scalar("evaluation/episodic_length", infos[i]["episode"]["l"], global_step)

                writer.add_scalar("evaluation/thruster", infos[i]["thruster"], global_step)
                writer.add_scalar("evaluation/thruster_square", infos[i]["thruster_square"], global_step)
                writer.add_scalar("evaluation/thruster_mean", infos[i]["thruster"] / infos[i]["episode"]["l"],
                                  global_step)
                writer.add_scalar("evaluation/total_distance_moved", infos[i]["total_distance_moved"], global_step)
                writer.add_scalar("evaluation/velocity=total_distance_moved/t",
                                  infos[i]["total_distance_moved"] / (0.1 * infos[i]["episode"]["l"]), global_step)


        if np.any(next_done_):

            ind = np.where(next_done_)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], step + 1)

        if not np.any(unfinished):
            break


    # print("step:", step)
    if haha != True:
        raise ValueError("no episode info")
    return episode_return.mean(), episode_length.mean(),time.time()-t