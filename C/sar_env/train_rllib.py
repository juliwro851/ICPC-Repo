from __future__ import annotations
import argparse
import csv
import json
import ray
from pathlib import Path
from typing import Dict, Any, Optional

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec

from sar_env import env_creator as base_env_creator, SARParallelEnv
try:
    from sar_env import ROLE_RESCUER, ROLE_FIREFIGHTER
except Exception:
    ROLE_RESCUER, ROLE_FIREFIGHTER = "rescuer", "firefighter"
try:
    from utils import save_frames_gif
except Exception:
    import imageio.v2 as imageio
    from pathlib import Path as _Path
    def save_frames_gif(frames, base_filename, fps=8, folder="gifs"):
        _Path(folder).mkdir(parents=True, exist_ok=True)
        out = _Path(folder) / f"{base_filename}.gif"
        imageio.mimsave(out, frames, fps=fps)
        return str(out)


class HeterogeneousCallback(DefaultCallbacks):    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        if info and isinstance(info, dict):
            for key in ["episode_success", "rescues_done", "victims_remaining", 
                       "rubble_cleared", "wall_hits", "moves", "noop"]:
                if key in info:
                    episode.custom_metrics[key] = info[key]
            
            total_rescues = sum(v for k, v in info.items() if k.endswith("_rescues"))
            total_clears = sum(v for k, v in info.items() if k.endswith("_clears"))
            
            episode.custom_metrics["total_agent_rescues"] = total_rescues
            episode.custom_metrics["total_agent_clears"] = total_clears
            
            rescuer_count = sum(1 for k, v in info.items() if k.endswith("_role") and v == ROLE_RESCUER)
            ff_count = sum(1 for k, v in info.items() if k.endswith("_role") and v == ROLE_FIREFIGHTER)
            
            if rescuer_count > 0:
                episode.custom_metrics["rescuer_count"] = rescuer_count
            if ff_count > 0:
                episode.custom_metrics["firefighter_count"] = ff_count


def make_env(env_config):
    base = base_env_creator(**env_config)
    return ParallelPettingZooEnv(base)


def get_policy_specs_and_mapping(env_config: Dict[str, Any]) -> tuple:    
    dummy_env = base_env_creator(**env_config)
    obs, _ = dummy_env.reset()
    
    policies = {}
    
    if env_config.get("heterogeneous", False):
        rescuer_agent = None
        firefighter_agent = None
        
        for agent_id in dummy_env.agents:
            role = getattr(dummy_env, "agent_roles", {}).get(agent_id, ROLE_RESCUER)
            if role == ROLE_RESCUER and rescuer_agent is None:
                rescuer_agent = agent_id
            elif role == ROLE_FIREFIGHTER and firefighter_agent is None:
                firefighter_agent = agent_id
        
        if rescuer_agent:
            rescuer_obs_space = dummy_env.observation_space(rescuer_agent)
            rescuer_action_space = dummy_env.action_space(rescuer_agent)
            
            policies["rescuer"] = PolicySpec(
                observation_space=rescuer_obs_space,
                action_space=rescuer_action_space,
                config={"model": {"fcnet_hiddens": [128, 128, 64]}}
            )
        
        if firefighter_agent:
            ff_obs_space = dummy_env.observation_space(firefighter_agent)
            ff_action_space = dummy_env.action_space(firefighter_agent)
            
            policies["firefighter"] = PolicySpec(
                observation_space=ff_obs_space,
                action_space=ff_action_space,
                config={"model": {"fcnet_hiddens": [128, 128, 64]}}
            )
        
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            try:
                agent_idx = int(agent_id.split("_")[-1])
            except (ValueError, IndexError):
                agent_idx = 0
            
            n_rescuers = env_config.get("n_rescuers")
            if n_rescuers is not None:
                return "rescuer" if agent_idx < n_rescuers else "firefighter"
            else:
                return "rescuer" if (agent_idx % 2 == 0) else "firefighter"
        
        policies_to_train = ["rescuer", "firefighter"]
        
    else:
        agent_obs_space = dummy_env.observation_space(dummy_env.agents[0])
        agent_action_space = dummy_env.action_space(dummy_env.agents[0])
        
        policies["shared_policy"] = PolicySpec(
            observation_space=agent_obs_space,
            action_space=agent_action_space,
            config={"model": {"fcnet_hiddens": [128, 128, 64]}}
        )
        
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return "shared_policy"
        
        policies_to_train = ["shared_policy"]
    
    dummy_env.close()
    return policies, policy_mapping_fn, policies_to_train


def save_training_config(args, filename: str):
    config = {
        "grid_size": args.grid_size,
        "n_agents": args.agents,
        "n_victims": args.victims,
        "rubble_density": args.rubble,
        "max_steps": args.steps,
        "heterogeneous": args.hetero,
        "seed": args.seed,
        "training_iterations": args.iters,
        "n_rescuers": args.rescuers,
        "n_firefighters": args.firefighters,
        "algo": "PPO",
        "framework": "torch",
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--victims", type=int, default=10)
    parser.add_argument("--rubble", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hetero", action="store_true",
                        help="Use heterogeneous mode (different roles).")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--rescuers", type=int, default=None, 
                        help="Number of rescuers (hetero mode only).")
    parser.add_argument("--firefighters", type=int, default=None, 
                        help="Number of firefighters (hetero mode only).")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size")

    args = parser.parse_args()

    if args.hetero and (args.rescuers is not None or args.firefighters is not None):
        if args.rescuers is None:
            args.rescuers = 0
        if args.firefighters is None:
            args.firefighters = 0
        if args.rescuers + args.firefighters != args.agents:
            print(f"WARNING: Agent count changed from {args.agents} to {args.rescuers + args.firefighters}")
            args.agents = args.rescuers + args.firefighters

    env_name = "SAR_PZ_ENV_UNIFIED"
    register_env(env_name, make_env)

    env_config = dict(
        grid_size=args.grid_size,
        n_agents=args.agents,
        n_victims=args.victims,
        rubble_density=args.rubble,
        max_steps=args.steps,
        heterogeneous=args.hetero,
        seed=args.seed,
        n_rescuers=args.rescuers,
        n_firefighters=args.firefighters,
    )

    policies, policy_mapping_fn, policies_to_train = get_policy_specs_and_mapping(env_config)

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config)
        .framework("torch")
        .callbacks(HeterogeneousCallback)
        .env_runners(num_env_runners=0)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
        num_env_runners=env_config.get("num_env_runners", 2),  
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            train_batch_size=args.batch_size,
            model={"vf_share_layers": False}
        )
    )

    if args.hetero:
        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )

    seed_str = args.seed if args.seed is not None else "random"
    mode = "hetero" if args.hetero else "homo"
    
    run_dir = Path("runs") / f"{mode}_{seed_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = run_dir / "params.json"
    save_training_config(args, str(config_file))
    
    results_file = run_dir / f"results_{mode}_{seed_str}.csv"
    csv_file = open(results_file, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "reward_mean", "episode_len_mean", "episode_success_rate", "mode"])

    def _extract_metric(res: dict, primary: str, fallbacks=(), default=None):
        if primary in res and res[primary] is not None:
            return res[primary]
        for container in ("env_runners", "sampler_results", "evaluation"):
            d = res.get(container, {})
            if isinstance(d, dict) and primary in d and d[primary] is not None:
                return d[primary]
        for name in fallbacks:
            if name in res and res[name] is not None:
                return res[name]
            for container in ("env_runners", "sampler_results", "evaluation"):
                d = res.get(container, {})
                if isinstance(d, dict) and name in d and d[name] is not None:
                    return d[name]
        return default

    print(f"Building {mode.upper()} PPO algorithm...")
    if args.hetero:
        print(f"  - Rescuers: {args.rescuers or args.agents//2}")
        print(f"  - Firefighters: {args.firefighters or args.agents - (args.rescuers or args.agents//2)}")
    
    algo = config.build()

    gif_dir = run_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    def _get_policy_for_agent(agent_id: str) -> str:
        if not args.hetero:
            return "default_policy"
        try:
            agent_idx = int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            agent_idx = 0
        
        if args.rescuers is not None:
            return "rescuer" if agent_idx < args.rescuers else "firefighter"
        else:
            return "rescuer" if (agent_idx % 2 == 0) else "firefighter"

    print(f"\nStarting {args.iters} training iterations...")
    for i in range(args.iters):
        result = algo.train()

        reward_mean = _extract_metric(result, "episode_reward_mean")
        episode_len_mean = _extract_metric(result, "episode_len_mean")
        
        custom_metrics = result.get("custom_metrics", {})
        success_rate = custom_metrics.get("episode_success", 0.0)

        rew_str = f"{reward_mean:.3f}" if reward_mean is not None else "N/A"
        elen_str = f"{episode_len_mean:.1f}" if episode_len_mean is not None else "N/A"
        succ_str = f"{success_rate:.3f}" if success_rate is not None else "N/A"
        
        print(f"[{mode.upper()}] iter {i+1}/{args.iters}: "
              f"reward={rew_str}, ep_len={elen_str}, success={succ_str}")
        
        csv_writer.writerow([i + 1, reward_mean, episode_len_mean, success_rate, mode])
        csv_file.flush()

        if (i + 1) % max(1, args.iters // 5) == 0 or i == args.iters - 1: 
            print(f"  Creating visualization for iter {i+1}...")
            
            viz_env = SARParallelEnv(**env_config)
            viz_env.iteration_idx = i               
            obs = viz_env.reset(options={"iter_idx": i, "episode_idx": 1})

            frames = []
            
            try:
                frame = viz_env.render_array()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"[WARN] Render error: {e}")

            for step in range(env_config["max_steps"]):
                actions = {}
                for agent_id, ob in obs.items():
                    policy_id = _get_policy_for_agent(agent_id)
                    action, _, _ = algo.get_policy(policy_id).compute_single_action(ob)
                    actions[agent_id] = int(action)

                obs, dones, truncs = viz_env.step(actions)

                try:
                    frame = viz_env.render_array()
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    pass

                if dones.get("__all__", False) or truncs.get("__all__", False):
                    break
    
            if frames:
                gif_path = gif_dir / f"train_{mode}_iter_{i+1:03d}.gif"
                print(f"  Saving GIF with {len(frames)} frames to {gif_path}")
                save_frames_gif(frames, base_filename=gif_path.stem, fps=8, folder=str(gif_dir))
            else:
                print("  [WARN] No frames captured; GIF will not be saved.")


            viz_env.close()

    csv_file.close()
    algo.stop()

    print(f"\nTraining completed! Results saved to: {run_dir}")
    print(f"  - Configuration: {config_file}")
    print(f"  - Training data: {results_file}")
    print(f"  - Visualizations: {gif_dir}")


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(include_dashboard=False, ignore_reinit_error=True)
    main()
