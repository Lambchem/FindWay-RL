"""
eval.py —— 加载训练好的 PPO checkpoint，评测模型性能并可视化。

用法：
    python eval.py                          # 批量评测 + 自动 pygame 演示
    python eval.py --no-render              # 仅批量评测，不打开窗口
    python eval.py --episodes 2000          # 指定批量评测局数
    python eval.py --ckpt my_model.pt       # 指定 checkpoint 路径
    python eval.py --seed 42                # 固定随机种子（可复现）
    python eval.py --greedy                 # 贪婪策略（argmax，不采样）
"""

import os, sys, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

# ── pygame 按需导入（--no-render 时不需要）──────────────────────────────────
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# ─────────────────────────────────────────────────────────────────────────────
# 与 train.py 保持完全一致的环境和模型定义
# ─────────────────────────────────────────────────────────────────────────────

class GridWorldRandom:
    """单个随机迷宫环境（BFS 保证可达）"""
    def __init__(self, size=10, obstacle_p=0.65, max_steps=50, seed=None):
        self.size        = size
        self.obstacle_p  = obstacle_p
        self.max_steps   = max_steps
        self.rng         = np.random.RandomState(seed)
        self.reset()

    def _bfs_reachable(self, grid):
        n, s, g = self.size, (0, 0), (self.size - 1, self.size - 1)
        if grid[s] or grid[g]:
            return False
        q, seen, dirs = deque([s]), {s}, [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            x, y = q.popleft()
            if (x, y) == g:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and not grid[nx, ny]:
                    seen.add((nx, ny)); q.append((nx, ny))
        return False

    def _sample_map(self):
        n = self.size
        while True:
            grid = (self.rng.rand(n, n) < self.obstacle_p).astype(np.int32)
            grid[0, 0] = grid[n-1, n-1] = 0
            if self._bfs_reachable(grid):
                return grid

    def reset(self):
        self.grid      = self._sample_map()
        self.agent_pos = [0, 0]
        self.steps     = 0
        return self._get_obs()

    def _get_obs(self):
        ax, ay = self.agent_pos
        gx, gy = self.size - 1, self.size - 1
        return np.concatenate([
            np.array([ax, ay, gx, gy], dtype=np.float32) / self.size,
            self.grid.flatten().astype(np.float32),
        ])

    def step(self, action):
        self.steps += 1
        x, y   = self.agent_pos
        nx, ny = x, y
        if action == 0: nx -= 1
        elif action == 1: nx += 1
        elif action == 2: ny -= 1
        elif action == 3: ny += 1

        reward = -0.01
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[nx, ny] == 1:
            reward = -0.05; nx, ny = x, y

        self.agent_pos = [nx, ny]
        done = False
        if self.agent_pos == [self.size - 1, self.size - 1]:
            reward = 1.0; done = True
        elif self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.pi = nn.Linear(256, act_dim)
        self.v  = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.pi(x), self.v(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 批量评测
# ─────────────────────────────────────────────────────────────────────────────

def batch_eval(model, device, n_episodes=1000, size=10, obstacle_p=0.65,
               max_steps=50, greedy=False, base_seed=random.randint(0, 2**31-1)):
    """
    跑 n_episodes 局，统计：
      - 成功率（到达终点的比例）
      - 平均总奖励（每局）
      - 平均步数（成功局）
      - 最短成功步数
    """
    model.eval()
    successes   = 0
    total_rews  = []
    success_steps = []

    print(f"\n{'─'*55}")
    print(f"  批量评测：{n_episodes} 局  |  greedy={greedy}")
    print(f"{'─'*55}")
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        env = GridWorldRandom(size=size, obstacle_p=obstacle_p,
                              max_steps=max_steps, seed=base_seed + ep)
        obs  = env.reset()
        done = False
        ep_rew = 0.0

        while not done:
            with torch.no_grad():
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                if greedy:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())

            obs, rew, done = env.step(action)
            ep_rew += rew

        reached_goal = (env.agent_pos == [env.size - 1, env.size - 1])
        if reached_goal:
            successes += 1
            success_steps.append(env.steps)
        total_rews.append(ep_rew)

    elapsed = time.perf_counter() - t0

    success_rate = successes / n_episodes * 100
    avg_rew      = float(np.mean(total_rews))
    avg_steps_ok = float(np.mean(success_steps)) if success_steps else float("nan")
    min_steps_ok = int(np.min(success_steps))     if success_steps else -1

    print(f"  评测用时       : {elapsed:.1f}s")
    print(f"  成功率         : {successes}/{n_episodes}  ({success_rate:.1f}%)")
    print(f"  平均总奖励     : {avg_rew:.3f}")
    print(f"  成功局平均步数 : {avg_steps_ok:.1f}")
    print(f"  成功局最短步数 : {min_steps_ok}")
    print(f"{'─'*55}\n")

    return success_rate, avg_rew


# ─────────────────────────────────────────────────────────────────────────────
# pygame 可视化
# ─────────────────────────────────────────────────────────────────────────────

class PygameRenderer:
    def __init__(self, n, cell_px=56, fps=8):
        self.n    = n
        self.cell = cell_px
        self.fps  = fps
        pygame.init()
        self.screen = pygame.display.set_mode((n * cell_px, n * cell_px))
        pygame.display.set_caption("GridWorld PPO – Eval")
        self.clock  = pygame.font.init() or pygame.time.Clock()
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("consolas", 14, bold=True)
        self.C = dict(
            bg    = (245, 245, 245),
            grid  = (210, 210, 210),
            wall  = (30,  30,  30),
            goal  = (50,  200, 100),
            agent = (60,  110, 230),
            trail = (180, 200, 255),
            text  = (50,  50,  50),
        )

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

    def draw(self, grid, agent_pos, trail, ep_idx, success, steps, ep_rew):
        c, n = self.cell, self.n
        self.screen.fill(self.C["bg"])

        # 障碍
        for x in range(n):
            for y in range(n):
                if grid[x, y] == 1:
                    pygame.draw.rect(self.screen, self.C["wall"],
                                     (y*c, x*c, c, c))

        # 轨迹
        for (tx, ty) in trail:
            pygame.draw.rect(self.screen, self.C["trail"],
                             (ty*c+4, tx*c+4, c-8, c-8))

        # 终点
        gn = n - 1
        pygame.draw.rect(self.screen, self.C["goal"],
                         (gn*c, gn*c, c, c))

        # Agent
        ax, ay = agent_pos
        pygame.draw.rect(self.screen, self.C["agent"],
                         (ay*c+2, ax*c+2, c-4, c-4))

        # 网格线
        for i in range(n + 1):
            pygame.draw.line(self.screen, self.C["grid"], (0, i*c), (n*c, i*c), 1)
            pygame.draw.line(self.screen, self.C["grid"], (i*c, 0), (i*c, n*c), 1)

        # 信息文字
        status = "SUCCESS ✓" if success else f"step {steps}"
        color  = (0, 160, 60) if success else self.C["text"]
        info   = f"Ep {ep_idx}  |  {status}  |  rew={ep_rew:.2f}"
        surf   = self.font.render(info, True, color)
        self.screen.blit(surf, (6, 4))

        pygame.display.flip()
        self.clock.tick(self.fps)


def visual_eval(model, device, n_episodes=20, size=10, obstacle_p=0.65,
                max_steps=50, greedy=True, fps=8, base_seed=random.randint(0, 2**31-1)):
    if not HAS_PYGAME:
        print("pygame 未安装，跳过可视化。")
        return

    ren = PygameRenderer(size, cell_px=56, fps=fps)
    model.eval()

    for ep in range(n_episodes):
        env   = GridWorldRandom(size=size, obstacle_p=obstacle_p,
                                max_steps=max_steps, seed=base_seed + ep)
        obs   = env.reset()
        done  = False
        trail = []
        ep_rew = 0.0

        while not done:
            ren.handle_events()
            trail.append(tuple(env.agent_pos))

            with torch.no_grad():
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                if greedy:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())

            obs, rew, done = env.step(action)
            ep_rew += rew
            success = (env.agent_pos == [env.size - 1, env.size - 1])
            ren.draw(env.grid, env.agent_pos, trail, ep + 1,
                     success and done, env.steps, ep_rew)

        # 结局停留 0.6s
        t_end = time.time() + 0.6
        while time.time() < t_end:
            ren.handle_events()
            ren.draw(env.grid, env.agent_pos, trail, ep + 1,
                     env.agent_pos == [env.size-1, env.size-1], env.steps, ep_rew)

    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="ppo_checkpoint_finetuned.pt", help="checkpoint 路径")
    p.add_argument("--episodes",   type=int,   default=1000,    help="批量评测局数")
    p.add_argument("--vis-eps",    type=int,   default=20,      help="可视化演示局数")
    p.add_argument("--size",       type=int,   default=10,      help="地图尺寸")
    p.add_argument("--obstacle-p", type=float, default=0.65,     help="障碍密度")
    p.add_argument("--max-steps",  type=int,   default=50,      help="每局最大步数")
    p.add_argument("--fps",        type=int,   default=8,       help="可视化帧率")
    p.add_argument("--greedy",     action="store_true",         help="贪婪策略（argmax）")
    p.add_argument("--no-render",  action="store_true",         help="跳过 pygame 可视化")
    p.add_argument("--seed",       type=int,   default=None,    help="numpy/torch 随机种子")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    obs_dim = 4 + args.size * args.size
    model   = ActorCritic(obs_dim, act_dim=4).to(device)

    if not os.path.exists(args.ckpt):
        sys.exit(f"[错误] 找不到 checkpoint：{args.ckpt}\n"
                 "请先运行 train.py 完成训练，或用 --ckpt 指定路径。")

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    trained_update = ckpt.get("update", "?")
    print(f"已加载 checkpoint（训练至 update={trained_update}）：{args.ckpt}")

    # ── 批量评测 ──────────────────────────────────────────────────────────────
    batch_eval(
        model, device,
        n_episodes  = args.episodes,
        size        = args.size,
        obstacle_p  = args.obstacle_p,
        max_steps   = args.max_steps,
        greedy      = args.greedy,
    )

    # ── pygame 可视化 ─────────────────────────────────────────────────────────
    if not args.no_render:
        print("开始 pygame 可视化演示（按 ESC 或关闭窗口退出）...")
        visual_eval(
            model, device,
            n_episodes  = args.vis_eps,
            size        = args.size,
            obstacle_p  = args.obstacle_p,
            max_steps   = args.max_steps,
            greedy      = args.greedy,
            fps         = args.fps,
        )
