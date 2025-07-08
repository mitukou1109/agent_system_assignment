import argparse
import os
import pickle

import genesis as gs
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="athletics")
    parser.add_argument("-l", "--log_dir", type=str, default="logs")
    parser.add_argument("-p", "--param_name", type=str, default="test")
    parser.add_argument("--ckpt", type=int, default=99)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"{args.log_dir}/{args.exp_name}/{args.param_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path, map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
