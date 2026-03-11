import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows/Conda 常见 OpenMP 冲突

import time
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
# torch.amp (新 API，替代 deprecated torch.cuda.amp)

CHECKPOINT_PATH = "ppo_checkpoint.pt"


# =========================
# 1) 单个随机障碍 GridWorld（保证可达）
# =========================
class GridWorldRandom:
    def __init__(self, size=10, obstacle_p=0.2, max_steps=250, seed=None):
        self.size = size
        self.obstacle_p = obstacle_p
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.reset()

    def _bfs_reachable(self, grid):
        n = self.size
        start = (0, 0)
        goal = (n - 1, n - 1)
        if grid[start] == 1 or grid[goal] == 1:
            return False

        q = deque([start])
        seen = {start}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen:
                    if grid[nx, ny] == 0:
                        seen.add((nx, ny))
                        q.append((nx, ny))
        return False

    def _sample_map(self):
        n = self.size
        while True:
            grid = (self.rng.rand(n, n) < self.obstacle_p).astype(np.int32)
            grid[0, 0] = 0
            grid[n - 1, n - 1] = 0
            if self._bfs_reachable(grid):
                return grid

    def reset(self):
        self.grid = self._sample_map()
        self.agent_pos = [0, 0]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        ax, ay = self.agent_pos
        gx, gy = self.size - 1, self.size - 1
        obs = np.concatenate([
            np.array([ax, ay, gx, gy], dtype=np.float32) / self.size,
            self.grid.flatten().astype(np.float32),
        ])
        return obs

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos
        nx, ny = x, y

        # 0 up, 1 down, 2 left, 3 right
        if action == 0:
            nx -= 1
        elif action == 1:
            nx += 1
        elif action == 2:
            ny -= 1
        elif action == 3:
            ny += 1

        reward = -0.01

        # 撞墙/撞障碍
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[nx, ny] == 1:
            reward = -0.05
            nx, ny = x, y

        self.agent_pos = [nx, ny]

        done = False
        if self.agent_pos == [self.size - 1, self.size - 1]:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done


# =========================
# 2) 全 numpy 向量化并行环境
#    step() 用 numpy broadcast 一次处理所有 N 个环境，彻底消除 Python for 循环，
#    让 GPU 不再等 CPU。
# =========================
class VecGridWorldFast:
    def __init__(self, n_envs, size=10, obstacle_p=0.2, max_steps=250, base_seed=1234):
        self.N          = n_envs
        self.size       = size
        self.obstacle_p = obstacle_p
        self.max_steps  = max_steps
        self.obs_dim    = 4 + size * size
        self._rngs      = [np.random.RandomState(base_seed + i) for i in range(n_envs)]
        self.grids = np.zeros((n_envs, size, size), dtype=np.int32)
        self.pos   = np.zeros((n_envs, 2),          dtype=np.int32)
        self.steps = np.zeros(n_envs,               dtype=np.int32)
        self.reset()

    # ------ 地图生成（BFS 保证可达）------
    def _bfs_ok(self, grid):
        n, s, e = self.size, (0, 0), (self.size - 1, self.size - 1)
        if grid[s] or grid[e]:
            return False
        q, seen, dirs = deque([s]), {s}, [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            x, y = q.popleft()
            if (x, y) == e:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and not grid[nx, ny]:
                    seen.add((nx, ny)); q.append((nx, ny))
        return False

    def _new_grid(self, i):
        rng, n, p = self._rngs[i], self.size, self.obstacle_p
        while True:
            g = (rng.rand(n, n) < p).astype(np.int32)
            g[0, 0] = g[n-1, n-1] = 0
            if self._bfs_ok(g):
                return g

    # ------ 批量 obs [N, obs_dim] ------
    def _obs(self):
        S = self.size
        ax = self.pos[:, 0].astype(np.float32) / S
        ay = self.pos[:, 1].astype(np.float32) / S
        gv = np.float32((S - 1) / S)
        pos_part  = np.stack([ax, ay,
                               np.full(self.N, gv, np.float32),
                               np.full(self.N, gv, np.float32)], axis=1)  # [N,4]
        grid_part = self.grids.reshape(self.N, -1).astype(np.float32)     # [N,S*S]
        return np.concatenate([pos_part, grid_part], axis=1)

    # ------ 公开接口 ------
    def reset(self):
        for i in range(self.N):
            self.grids[i] = self._new_grid(i)
        self.pos[:]   = 0
        self.steps[:] = 0
        return self._obs()

    def step(self, actions):
        """actions: int ndarray [N]，全批量无 Python 循环"""
        self.steps += 1
        x, y   = self.pos[:, 0].copy(), self.pos[:, 1].copy()
        nx, ny = x.copy(), y.copy()

        # 批量移动
        nx = np.where(actions == 0, x - 1, nx)
        nx = np.where(actions == 1, x + 1, nx)
        ny = np.where(actions == 2, y - 1, ny)
        ny = np.where(actions == 3, y + 1, ny)

        # 越界 & 障碍检测（先 clip 再查 grid，避免越界索引）
        nx_c = np.clip(nx, 0, self.size - 1)
        ny_c = np.clip(ny, 0, self.size - 1)
        ei   = np.arange(self.N)
        invalid = ((nx < 0) | (nx >= self.size) |
                   (ny < 0) | (ny >= self.size) |
                   (self.grids[ei, nx_c, ny_c] == 1))

        self.pos[:, 0] = np.where(invalid, x, nx)
        self.pos[:, 1] = np.where(invalid, y, ny)

        rews    = np.where(invalid, np.float32(-0.05), np.float32(-0.01))
        at_goal = (self.pos[:, 0] == self.size - 1) & (self.pos[:, 1] == self.size - 1)
        rews    = np.where(at_goal, np.float32(1.0), rews)
        dones   = at_goal | (self.steps >= self.max_steps)

        # 只重置完成的环境（BFS 仅对少量 done 环境调用）
        for i in np.where(dones)[0]:
            self.grids[i] = self._new_grid(i)
            self.pos[i]   = 0
            self.steps[i] = 0

        return self._obs(), rews, dones


# =========================
# 3) PPO 模型：共享层 + actor + critic
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, act_dim)
        self.v = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.pi(x), self.v(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v


# =========================
# 4) GAE(λ) ——在 GPU 上计算，避免 CPU 往返
# =========================
def compute_gae_gpu(rews, dones, values, last_values, gamma=0.99, lam=0.95):
    """
    rews/dones/values: [T, N] float32 GPU tensors
    last_values: [N] GPU tensor
    dones 已是 float32 (0/1)
    """
    T, N = rews.shape
    adv      = torch.zeros_like(rews)
    last_gae = torch.zeros(N, dtype=torch.float32, device=rews.device)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_v = last_values if t == T - 1 else values[t + 1]
        delta    = rews[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t]   = last_gae

    ret = adv + values
    return adv, ret


# =========================
# 5) Checkpoint 工具
# =========================
def save_checkpoint(path, model, optimizer, scaler, update, running_rew):
    torch.save({
        "update":               update,
        "running_rew":          running_rew,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
    }, path)
    print(f"  [ckpt] 已保存至 {path}  (update={update})")


def load_checkpoint(path, model, optimizer, scaler, device):
    """返回 (start_update, running_rew)；文件不存在则返回 (0, 0.0)"""
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model    .load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler   .load_state_dict(ckpt["scaler_state_dict"])
    start = ckpt.get("update",      0)
    r_rew = ckpt.get("running_rew", 0.0)
    print(f"  [ckpt] 从 {path} 恢复，继续自 update={start}")
    return start, r_rew


# =========================
# 6) PPO 训练（GPU rollout 缓冲 + AMP + checkpoint）
# =========================
def train_ppo_vec(
    size=10,
    obstacle_p=0.2,
    max_steps=250,
    n_envs=64,
    horizon=256,
    total_updates=800,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    epochs=4,
    minibatch_size=4096,
    save_every=100,          # 每隔多少 update 保存一次
    resume=True,             # 是否从上次断点继续
    device=None,
):
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")          # 仅 CUDA 开启混合精度
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 自动选最快卷积算法（对 MLP 也有小收益）
    print(f"Using device: {device}  |  AMP: {use_amp}")

    vec    = VecGridWorldFast(n_envs, size=size, obstacle_p=obstacle_p,
                               max_steps=max_steps, base_seed=1234)
    obs_np = vec.reset()                        # [N, obs_dim]  numpy
    obs_dim = 4 + size * size

    model     = ActorCritic(obs_dim, 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- 断点恢复 ----
    start_update = 0
    running_rew  = 0.0
    if resume:
        start_update, running_rew = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scaler, device
        )

    # ---- 预分配 GPU rollout 缓冲区（避免每步 new tensor）----
    obs_buf  = torch.zeros((horizon, n_envs, obs_dim), dtype=torch.float32, device=device)
    act_buf  = torch.zeros((horizon, n_envs),          dtype=torch.int64,   device=device)
    logp_buf = torch.zeros((horizon, n_envs),          dtype=torch.float32, device=device)
    rew_buf  = torch.zeros((horizon, n_envs),          dtype=torch.float32, device=device)
    done_buf = torch.zeros((horizon, n_envs),          dtype=torch.float32, device=device)
    val_buf  = torch.zeros((horizon, n_envs),          dtype=torch.float32, device=device)

    # 将初始观测放到 GPU，后续全程保持在 GPU
    obs_gpu = torch.from_numpy(obs_np).to(device, non_blocking=True)

    # 预分配 pinned-memory CPU 缓冲，加速异步 H2D 传输
    if device.type == "cuda":
        _obs_pin  = torch.empty(n_envs, obs_dim,  dtype=torch.float32).pin_memory()
        _rew_pin  = torch.empty(n_envs,           dtype=torch.float32).pin_memory()
        _done_pin = torch.empty(n_envs,           dtype=torch.float32).pin_memory()
    def _to_gpu(arr_f32, pin):
        pin.copy_(torch.from_numpy(arr_f32))
        return pin.to(device, non_blocking=True)

    log_interval  = 20
    # running_rew 每步累加 rews.mean()（已对 n_envs 求均值），
    # 所以分母只是 timestep 数，不再乘 n_envs
    steps_per_log = log_interval * horizon

    for upd in range(start_update + 1, total_updates + 1):
        t0 = time.perf_counter()

        # ===== 收集 rollout（GPU 缓冲，减少 CPU↔GPU 往返）=====
        model.eval()
        with torch.no_grad():
            for t in range(horizon):
                a, logp, v = model.act(obs_gpu)

                actions = a.cpu().numpy()                          # env step 仍在 CPU
                next_obs_np, rews, dones = vec.step(actions)

                obs_buf [t] = obs_gpu
                act_buf [t] = a
                logp_buf[t] = logp
                val_buf [t] = v
                if device.type == "cuda":
                    rew_buf [t] = _to_gpu(rews,                     _rew_pin)
                    done_buf[t] = _to_gpu(dones.astype(np.float32), _done_pin)
                    obs_gpu     = _to_gpu(next_obs_np,               _obs_pin)
                else:
                    rew_buf [t] = torch.from_numpy(rews)
                    done_buf[t] = torch.from_numpy(dones.astype(np.float32))
                    obs_gpu     = torch.from_numpy(next_obs_np)

                running_rew += float(rews.mean())

            # bootstrap value（已在 GPU，无需转换）
            last_v = model.forward(obs_gpu)[1]   # [N]

        # ===== GAE 全在 GPU 上计算 =====
        adv, ret = compute_gae_gpu(rew_buf, done_buf, val_buf, last_v, gamma=gamma, lam=lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ===== Flatten =====
        B        = horizon * n_envs
        obs_flat = obs_buf .reshape(B, obs_dim)
        act_flat = act_buf .reshape(B)
        logp_old = logp_buf.reshape(B)
        adv_flat = adv     .reshape(B)
        ret_flat = ret     .reshape(B)

        # ===== PPO 更新（AMP + 每 epoch 重新 shuffle）=====
        model.train()
        for ep in range(epochs):
            # 每个 epoch 独立 shuffle，保证 mini-batch 多样性
            idx = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = idx[start : start + minibatch_size]

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, v = model(obs_flat[mb])
                    dist  = torch.distributions.Categorical(logits=logits)
                    logp  = dist.log_prob(act_flat[mb])
                    ratio = torch.exp(logp - logp_old[mb])

                    surr1   = ratio * adv_flat[mb]
                    surr2   = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_flat[mb]
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss  = (ret_flat[mb] - v).pow(2).mean()
                    ent     = dist.entropy().mean()
                    loss    = pi_loss + vf_coef * v_loss - ent_coef * ent

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        # ===== 日志 =====
        if upd % log_interval == 0:
            elapsed     = time.perf_counter() - t0
            avg_rew     = running_rew / steps_per_log
            fps         = (log_interval * horizon * n_envs) / elapsed
            print(f"Update {upd:4d}/{total_updates} | avg_step_rew={avg_rew:.4f} "
                  f"| {fps:,.0f} env-steps/s | {elapsed:.1f}s/log")
            running_rew = 0.0

        # ===== 定期 & 末尾保存 checkpoint =====
        if upd % save_every == 0 or upd == total_updates:
            save_checkpoint(CHECKPOINT_PATH, model, optimizer, scaler, upd, running_rew)

    return model


# =========================
# 7) pygame 演示（单环境）
# =========================
class PygameRenderer:
    def __init__(self, n, cell_px=48, fps=12):
        self.n = n
        self.cell = cell_px
        self.fps = fps
        self.w = n * cell_px
        self.h = n * cell_px

        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("GridWorld PPO (pygame)")
        self.clock = pygame.time.Clock()

        self.C_BG = (245, 245, 245)
        self.C_GRID = (210, 210, 210)
        self.C_WALL = (25, 25, 25)
        self.C_GOAL = (60, 200, 120)
        self.C_AGENT = (70, 120, 240)

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit

    def draw(self, grid, agent_pos):
        self.screen.fill(self.C_BG)

        # walls
        for x in range(self.n):
            for y in range(self.n):
                if grid[x, y] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.C_WALL,
                        pygame.Rect(y * self.cell, x * self.cell, self.cell, self.cell),
                    )

        # goal
        gx, gy = self.n - 1, self.n - 1
        pygame.draw.rect(
            self.screen,
            self.C_GOAL,
            pygame.Rect(gy * self.cell, gx * self.cell, self.cell, self.cell),
        )

        # agent
        ax, ay = agent_pos
        pygame.draw.rect(
            self.screen,
            self.C_AGENT,
            pygame.Rect(ay * self.cell, ax * self.cell, self.cell, self.cell),
        )

        # grid lines
        for i in range(self.n + 1):
            pygame.draw.line(self.screen, self.C_GRID, (0, i * self.cell), (self.w, i * self.cell), 1)
            pygame.draw.line(self.screen, self.C_GRID, (i * self.cell, 0), (i * self.cell, self.h), 1)

        pygame.display.flip()
        self.clock.tick(self.fps)


def demo_pygame(model, size=10, obstacle_p=0.2, max_steps=250, episodes=20, fps=12, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldRandom(size=size, obstacle_p=obstacle_p, max_steps=max_steps, seed=999)
    ren = PygameRenderer(size, cell_px=48, fps=fps)

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            ren.handle_events()
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, _, done = env.step(action)
            ren.draw(env.grid, env.agent_pos)

        # 每局结束停一下
        t_end = time.time() + 0.35
        while time.time() < t_end:
            ren.handle_events()
            ren.draw(env.grid, env.agent_pos)

    pygame.quit()


# =========================
# 8) main
# =========================
if __name__ == "__main__":
    SIZE = 10
    OB_P = 0.20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 允许 TF32，A100/3090 等卡上矩阵乘法更快
        torch.set_float32_matmul_precision("high")

    model = train_ppo_vec(
        size=SIZE,
        obstacle_p=OB_P,
        max_steps=250,
        n_envs=1024,
        horizon=512,
        total_updates=1200,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=4,
        minibatch_size=8192,
        save_every=100,      # 每 100 次 update 自动保存
        resume=True,         # True = 检测上次 checkpoint 并继续；False = 从头训练
        device=device,
    )

    print("训练完成：开始 pygame 演示（关闭窗口退出）")
    demo_pygame(model, size=SIZE, obstacle_p=OB_P, episodes=30, fps=14, device=device)