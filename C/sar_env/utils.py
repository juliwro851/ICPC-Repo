from __future__ import annotations
from typing import Dict
import random
import imageio
import numpy as np

def seed_everything(seed: int | None) -> None:
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)

CELL_EMPTY  = 0
CELL_WALL   = 1
CELL_RUBBLE = 2
CELL_VICTIM = 3
CELL_BASE   = 4  

CELL_CHANS = 6

SYMBOLS = {
    CELL_EMPTY: ".",
    CELL_WALL:  "#",
    CELL_RUBBLE:"R",
    CELL_VICTIM:"V",
    CELL_BASE:  "B",
}

ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT = 1, 2, 3, 4
MOVE_DELTAS = {
    ACT_UP:    (-1, 0),
    ACT_DOWN:  ( 1, 0),
    ACT_LEFT:  ( 0,-1),
    ACT_RIGHT: ( 0, 1),
}

def save_episode_gif(env, filename="epizod.gif", max_steps=300, fps=10, folder="gify_epizodow"):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
    base, ext = os.path.splitext(filename)
    if ext.lower() not in (".gif", ""):
        ext = ".gif"
    if not base:
        base = "epizod"

    candidate = os.path.join(folder, f"{base}{ext or '.gif'}")
    if os.path.exists(candidate):
        i = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{i:03d}{ext or '.gif'}")
            if not os.path.exists(candidate):
                break
            i += 1
    filepath = candidate
    folder_to_use = os.path.dirname(filepath) or folder
    if not os.path.exists(folder_to_use):
        os.makedirs(folder_to_use, exist_ok=True)

    frames = []
    printed = False
    obs, infos = env.reset()
    done = {aid: False for aid in env.agents}
    step = 0

    while not all(done.values()) and step < max_steps:
        frame = env.render_array()
        if not printed:
            printed = True
            try:
                print(f"[GIF] frame shape: {getattr(frame, 'shape', None)}")
            except Exception:
                pass  
        frames.append(frame)

        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
        done = {aid: term.get(aid, False) or trunc.get(aid, False) for aid in env.agents}
        step += 1

    if not frames:
        print(f"[GIF] No frames for {filepath}")
        return
    imageio.mimsave(filepath, frames, duration=1.0/fps)




def gen_map(n: int, rubble_density: float, n_victims: int, rng: np.random.Generator
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert n >= 5
    grid = np.full((n, n), CELL_EMPTY, dtype=np.int32)
    grid[0, :] = CELL_WALL; grid[-1, :] = CELL_WALL
    grid[:, 0] = CELL_WALL; grid[:, -1] = CELL_WALL

    rubble_levels = np.zeros((n, n), dtype=np.int32)
    empties = np.argwhere(grid == CELL_EMPTY)
    rng.shuffle(empties)
    for (i, j) in empties:
        if rng.random() < rubble_density and 0 < i < n-1 and 0 < j < n-1:
            grid[i, j] = CELL_RUBBLE
            rubble_levels[i, j] = int(rng.integers(1, 4))  # 1..3

    candidates = np.argwhere(grid != CELL_WALL)
    rng.shuffle(candidates)
    victims_mask = np.zeros((n, n), dtype=bool)
    placed = 0
    for i, j in candidates:
        if placed >= n_victims:
            break
        if 0 < i < n-1 and 0 < j < n-1:
            victims_mask[i, j] = True
            placed += 1

    return grid, rubble_levels, victims_mask

def render_ascii(grid: np.ndarray, rubble_levels: np.ndarray, victims_mask: np.ndarray,
                 agent_positions: Dict[str, tuple[int,int]]) -> str:
    n = grid.shape[0]
    canvas = [[SYMBOLS.get(int(grid[i, j]), ".") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if victims_mask[i, j]:
                canvas[i][j] = "V"
            if int(grid[i, j]) == CELL_RUBBLE and rubble_levels[i, j] > 0:
                canvas[i][j] = "R"
            elif int(grid[i, j]) == CELL_RUBBLE and rubble_levels[i, j] == 0 and not victims_mask[i, j]:
                canvas[i][j] = "."
    for _, (r, c) in agent_positions.items():
        canvas[r][c] = "A"
    return "\n".join("".join(row) for row in canvas)


def save_frames_gif(frames, base_filename="train_episode", fps=10, folder="gify_epizodow"):

    import os, imageio
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        print("[GIF] No frames for (save_frames_gif).")
        return
    os.makedirs(folder, exist_ok=True)
    base = base_filename or "train_episode"
    ext = ".gif"
    candidate = os.path.join(folder, f"{base}{ext}")
    if os.path.exists(candidate):
        i = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{i:03d}{ext}")
            if not os.path.exists(candidate):
                break
            i += 1
    try:
        imageio.mimsave(candidate, frames, duration=1.0/max(fps,1))
    except Exception as e:
        print(f"[GIF] Saving failed GIF (save_frames_gif): {e}")
