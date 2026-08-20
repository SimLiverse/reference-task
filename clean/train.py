"""Entrypoint. `python train.py --task G1-Flat-v0 --num_envs 4096`.

The seed is read from the command line and not chosen here. The platform
assigns it and passes `--seed N` (ADR 021): a seed the script picks for itself
is a seed nobody recorded, and a run whose seed was never recorded cannot
belong to a seed family.
"""

import argparse

import gymnasium as gym

from g1_flat_env_cfg import G1FlatEnvCfg

gym.register(
    id="G1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1FlatEnvCfg},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="G1-Flat-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=1500)
    args = parser.parse_args()

    env = gym.make(args.task, num_envs=args.num_envs)
    if args.seed is not None:
        env.reset(seed=args.seed)

    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, log_dir="logs", device="cuda:0")
    runner.learn(num_learning_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
