from __future__ import annotations
import argparse
from sar_env import env_creator
from utils import save_episode_gif

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=11)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--victims", type=int, default=6)
    parser.add_argument("--rubble", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hetero", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)  # ile epizodów nagrać
    args = parser.parse_args()

    for ep in range(args.episodes):
        env = env_creator(grid_size=args.grid_size, n_agents=args.agents, n_victims=args.victims,
                          rubble_density=args.rubble, max_steps=args.steps, heterogeneous=args.hetero)
        # Tu automatyczne nagrywanie każdego epizodu:
        save_episode_gif(env, filename=f"gify_epizodow/epizod_{ep+1}.gif", max_steps=args.steps, folder="gify_epizodow")
        print(f"Nagrano epizod {ep+1}")

if __name__ == "__main__":
    main()

