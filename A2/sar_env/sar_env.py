from __future__ import annotations
import numpy as np
import csv
import os
from typing import Optional, Dict, Tuple
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

try:
    from .utils import (
        CELL_EMPTY, CELL_WALL, CELL_RUBBLE, CELL_VICTIM,
        ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, MOVE_DELTAS,
        gen_map,
    )
except ImportError:
    from utils import (
        CELL_EMPTY, CELL_WALL, CELL_RUBBLE, CELL_VICTIM,
        ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, MOVE_DELTAS,
        gen_map,
    )

ACT_NOOP   = 0
ACT_UP     = 1
ACT_DOWN   = 2
ACT_LEFT   = 3
ACT_RIGHT  = 4
ACT_CLEAR  = 5
ACT_RESCUE = 6

ROLE_RESCUER     = "rescuer"
ROLE_FIREFIGHTER = "firefighter" 


class SARParallelEnv(ParallelEnv):
    metadata = {"name": "sar_pz_unified", "render_modes": ["ansi"], "is_parallel": True}

    def __init__(
        self,
        grid_size: int = 20,
        n_agents: int = 4,
        n_victims: int = 10,
        rubble_density: float = 0.2,
        max_steps: int = 1000,
        heterogeneous: bool = False,  
        seed: Optional[int] = None,
        n_rescuers: Optional[int] = None,
        n_firefighters: Optional[int] = None,
    ):
        super().__init__()
        assert grid_size >= 7
        assert n_agents >= 2
        assert 0 <= rubble_density <= 1
        
        self.grid_size = grid_size
        self.n_victims = n_victims
        self.rubble_density = rubble_density
        self.max_steps = max_steps
        self.heterogeneous = heterogeneous
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.obs_radius = 2
        
        if self.heterogeneous:
            self.n_rescuers = n_rescuers
            self.n_firefighters = n_firefighters
            if (n_rescuers is not None) or (n_firefighters is not None):
                nr = n_rescuers or 0
                nf = n_firefighters or 0
                self.n_agents = nr + nf
            else:
                self.n_agents = n_agents
                self.n_rescuers = n_agents // 2
                self.n_firefighters = n_agents - self.n_rescuers
        else:
            self.n_agents = n_agents
            self.n_rescuers = None
            self.n_firefighters = None

        if self.heterogeneous:
            obs_dim = 77
        else:
            obs_dim = 75
            
        self.observation_spaces = {
            f"agent_{i}": spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.n_agents)
        }

        self.action_spaces = {}
        self.allowed_actions = {}

        self.grid = None
        self.rubble_levels = None
        self.victims_mask = None
        self.agents = []
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self.agent_roles: Dict[str, str] = {}
        self.stats = {}
        self.steps = 0

    def render_array(self):
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
        import numpy as np

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.grid, vmin=0, vmax=2, interpolation="nearest", cmap="gray")

        vy, vx = np.where(self.victims_mask)
        if len(vx) > 0:
            ax.scatter(
                vx, vy, marker="s", s=200,
                edgecolor="black", linewidths=0.5,
                facecolor="yellow", alpha=0.95,
                label="victim"
            )

        if self.heterogeneous:
            rescuer_color = "#d62728"    
            firefighter_color = "#1f77b4" 
            
            for agent, (x, y) in self.agent_positions.items():
                role = self.agent_roles.get(agent, ROLE_RESCUER)
                color = rescuer_color if role == ROLE_RESCUER else firefighter_color
                ax.scatter(x, y, marker='o', s=140,
                          facecolor=color, edgecolor="black", linewidths=0.6,
                          label=role if agent == self.agents[0] else None)
        else:
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
            
            for idx, (agent, (x, y)) in enumerate(self.agent_positions.items()):
                ax.scatter(x, y, marker='o', s=140,
                          facecolor=palette[idx % len(palette)],
                          edgecolor="black", linewidths=0.6)

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        fig.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        return np.array(image)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Generuj mapę
        self.grid, self.rubble_levels, self.victims_mask = gen_map(
            self.grid_size, self.rubble_density, self.n_victims, self._rng
        )

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.steps = 0

        self.stats = {
            "rescues_done": 0,
            "rescues_failed": 0,
            "victims_remaining": self.n_victims,
            "rubble_cleared": 0,
            "clear_attempts": 0,
            "wall_hits": 0,
            "rubble_hits": 0,
            "steps_taken": 0,
            "moves": 0,
            "noop": 0,
            "episode_success": 0,
        }

        self.agent_roles = {}
        self.allowed_actions = {}
        
        if self.heterogeneous:
            for i, agent in enumerate(self.agents):
                if i < (self.n_rescuers or (self.n_agents // 2)):
                    self.agent_roles[agent] = ROLE_RESCUER
                    self.allowed_actions[agent] = [ACT_NOOP, ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_RESCUE]
                else:
                    self.agent_roles[agent] = ROLE_FIREFIGHTER
                    self.allowed_actions[agent] = [ACT_NOOP, ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_CLEAR]
                    
            self.action_spaces = {
                agent: spaces.Discrete(len(self.allowed_actions[agent]))
                for agent in self.agents
            }
        else:
            all_actions = [ACT_NOOP, ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_CLEAR, ACT_RESCUE]
            for agent in self.agents:
                self.agent_roles[agent] = "generalist"  
                self.allowed_actions[agent] = all_actions
                
            self.action_spaces = {
                agent: spaces.Discrete(7) for agent in self.agents
            }

        self.agent_positions = {}
        empty_cells = list(zip(*np.where(self.grid == CELL_EMPTY)))
        self._rng.shuffle(empty_cells)
        for i, agent in enumerate(self.agents):
            pos = empty_cells[i]
            self.agent_positions[agent] = (pos[1], pos[0])  

        obs = {a: self._make_obs_for(a) for a in self.agents}
        infos = {a: {"role": self.agent_roles[a]} for a in self.agents}
        return obs, infos

    def step(self, actions):
        STEP_PENALTY = -0.003
        NOOP_PENALTY = -0.05
        MOVE_WALL_PENALTY = -1.0
        MOVE_RUBBLE_TOUCH_PENALTY = -0.3
        CLEAR_BASE_REWARD = 1.0
        CLEAR_OPEN_PATH_BONUS = 2.5
        FAILED_CLEAR_PENALTY = -0.4
        RESCUE_REWARD = 10.0

        self.steps += 1
        self.stats["steps_taken"] += 1

        rewards = {a: STEP_PENALTY for a in self.agents}
        dones = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for agent, local_action in actions.items():
            try:
                act = self.allowed_actions[agent][int(local_action)]
            except (IndexError, KeyError):
                act = ACT_NOOP

            x, y = self.agent_positions[agent]
            opened_path_this_step = False

            if act in [ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT]:
                self.stats["moves"] += 1
                dx, dy = MOVE_DELTAS[act]
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell = self.grid[ny, nx]
                    if cell == CELL_WALL:
                        self.stats["wall_hits"] += 1
                        rewards[agent] += MOVE_WALL_PENALTY
                    elif cell == CELL_RUBBLE:
                        self.stats["rubble_hits"] += 1
                        rewards[agent] += MOVE_RUBBLE_TOUCH_PENALTY
                    elif cell == CELL_EMPTY:
                        self.agent_positions[agent] = (nx, ny)
                else:
                    self.stats["wall_hits"] += 1
                    rewards[agent] += MOVE_WALL_PENALTY

            elif act == ACT_NOOP:
                self.stats["noop"] += 1
                rewards[agent] += NOOP_PENALTY

            elif act == ACT_CLEAR:
                self.stats["clear_attempts"] += 1
                cleared = False
                
                if self.heterogeneous:
                    efficiency = 2 if self.agent_roles[agent] == ROLE_FIREFIGHTER else 1
                else:
                    efficiency = 1
                
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.grid[ny, nx] == CELL_RUBBLE:
                            victim_nearby = False
                            for vx, vy in [(-1,0), (1,0), (0,-1), (0,1), (0,0)]:
                                vnx, vny = nx + vx, ny + vy
                                if 0 <= vnx < self.grid_size and 0 <= vny < self.grid_size:
                                    if self.grid[vny, vnx] == CELL_VICTIM:
                                        victim_nearby = True
                                        break

                            self.rubble_levels[ny, nx] -= efficiency
                            if self.rubble_levels[ny, nx] <= 0:
                                self.grid[ny, nx] = CELL_EMPTY
                                opened_path_this_step = True

                            if victim_nearby:
                                rewards[agent] += CLEAR_OPEN_PATH_BONUS  

                            self.stats["rubble_cleared"] += 1
                            rewards[agent] += CLEAR_BASE_REWARD
                            cleared = True
                            break
                            
                if not cleared:
                    rewards[agent] += FAILED_CLEAR_PENALTY

            elif act == ACT_RESCUE:
                if self.victims_mask[y, x]:
                    self.victims_mask[y, x] = False
                    self.stats["rescues_done"] += 1
                    self.stats["victims_remaining"] -= 1
                    rewards[agent] += RESCUE_REWARD
                else:
                    self.stats["rescues_failed"] += 1

            if opened_path_this_step:
                rewards[agent] += CLEAR_OPEN_PATH_BONUS


        if self.stats["victims_remaining"] == 0 or self.steps >= self.max_steps:
            self.stats["episode_success"] = int(self.stats["victims_remaining"] == 0)
            dones["__all__"] = True
            truncations = {a: self.steps >= self.max_steps for a in self.agents}
            truncations["__all__"] = self.steps >= self.max_steps
            
            seed_str = self.seed if self.seed is not None else "random"
            mode = "hetero" if self.heterogeneous else "homo"
            filename = f"episode_stats_{mode}_{seed_str}.csv"
            log_episode_stats(self.stats, filename)
        else:
            dones["__all__"] = False
            truncations["__all__"] = False

        obs = {a: self._make_obs_for(a) for a in self.agents}
        infos = {a: {"role": self.agent_roles[a]} for a in self.agents}
        return obs, rewards, dones, truncations, infos

    def _make_obs_for(self, agent):
        r = self.obs_radius
        H = W = 2 * r + 1
        x, y = self.agent_positions[agent]

        patch = np.ones((H, W), dtype=np.float32) * 0.5
        y0, y1 = max(0, y - r), min(self.grid_size, y + r + 1)
        x0, x1 = max(0, x - r), min(self.grid_size, x + r + 1)
        py0, py1 = r - (y - y0), r + (y1 - y)
        px0, px1 = r - (x - x0), r + (x1 - x)
        patch[py0:py1, px0:px1] = (self.grid[y0:y1, x0:x1].astype(np.float32) / 2.0)

        victims = np.zeros((H, W), dtype=np.float32)
        victims[py0:py1, px0:px1] = self.victims_mask[y0:y1, x0:x1].astype(np.float32)

        agents = np.zeros((H, W), dtype=np.float32)
        for other, (ox, oy) in self.agent_positions.items():
            if other == agent:
                continue
            if abs(ox - x) <= r and abs(oy - y) <= r:
                agents[r + (oy - y), r + (ox - x)] = 1.0

        base_obs = np.stack([patch, victims, agents], axis=0).reshape(-1).astype(np.float32)  # (75,)

        if self.heterogeneous:
            role = self.agent_roles.get(agent, ROLE_RESCUER)
            role_vec = np.array([1.0, 0.0], dtype=np.float32) if role == ROLE_RESCUER else np.array([0.0, 1.0], dtype=np.float32)
            return np.concatenate([base_obs, role_vec], axis=0) 
        else:
            return base_obs  


def log_episode_stats(stats, fname="episode_stats.csv"):
    file_exists = os.path.isfile(fname)
    with open(fname, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)


def env_creator(config=None, **kwargs):
    if config is None:
        config = {}
    config = {**config, **kwargs}

    return SARParallelEnv(
        grid_size=config.get("grid_size", 20),
        n_agents=config.get("n_agents", 4),
        n_victims=config.get("n_victims", 10),
        rubble_density=config.get("rubble_density", 0.2),
        max_steps=config.get("max_steps", 1000),
        heterogeneous=config.get("heterogeneous", False), 
        seed=config.get("seed", None),
        n_rescuers=config.get("n_rescuers", None),
        n_firefighters=config.get("n_firefighters", None),
    )