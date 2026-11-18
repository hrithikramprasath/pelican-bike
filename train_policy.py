"""Two-phase DAgger + Parallel MuJoCo PPO trainer for the Pelican Bicycle policy.

Phase 1: Computes exact linearized LQR bicycle gains via MuJoCo finite
differences and trains a pure-NumPy 2-hidden-layer MLP (``trained_policy.npz``)
via DAgger across randomized Figure-8 segments, pushes, carves, and wheelies.

Phase 2: Fine-tunes the Actor-Critic MLP directly on the nonlinear MuJoCo
dynamics using ``mujoco.batch.Batch`` PPO across high-speed Boost sprints,
sharp hairpins, heavy lateral pushes, wheelies, airborne landings, and
no-hands riding.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import mujoco
from mujoco import batch
import numpy as np
from scipy import linalg

import policy as policy_mod
import track

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(WORKSPACE_DIR, 'assets', 'pelican_bicycle', 'model.xml')

BIKE_PREFIX = 'pb_'
WHEEL_RADIUS = 0.330
WHEELBASE = 1.090
HEAD_ANGLE_RAD = math.radians(20.0)
CHAIN_RATIO = 2.0
SUBSTEPS = 10  # 10 * 0.002s = 0.02s (50 Hz control)


def build_training_model(*, with_posts: bool = False) -> mujoco.MjModel:
  """Builds clean ground-plane scene with pelican_bicycle attached."""
  spec = mujoco.MjSpec()
  spec.modelname = 'pelican_bicycle_train'
  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10.0

  asset_dir = os.path.join(WORKSPACE_DIR, 'assets', 'pelican_bicycle')
  spec.meshdir = os.path.join(asset_dir, 'meshes')
  spec.texturedir = os.path.join(asset_dir, 'textures')

  spec.worldbody.add_geom(
      name='ground_plane',
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[0, 0, 0.05],
      condim=6,
      friction=[1.2, 0.01, 0.001],
      group=1,
  )

  if with_posts:
    for i in range(20):
      th = 2.0 * math.pi * i / 20
      px = 10.6 * math.cos(th)
      py = 6.8 * math.sin(th)
      post = spec.worldbody.add_body(name=f'coastal_post_{i}', pos=[px, py, 0.0])
      post.add_geom(
          name=f'post_wood_{i}',
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=[0.0, 0.0, 0.42],
          size=[0.075, 0.42],
          group=1,
      )

  bike_spec = mujoco.MjSpec.from_file(MODEL_XML)
  bike_spec.worldbody.first_body().add_freejoint(name='free')
  spawn = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
  spec.attach(bike_spec, frame=spawn, prefix=BIKE_PREFIX)
  return spec.compile()


def compute_inner_lqr_table(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """Computes speed-scheduled inner balance LQR gains via MuJoCo FD."""
  d = mujoco.MjData(model)
  p = BIKE_PREFIX
  speeds = np.array([1.8, 2.5, 3.4, 4.5, 5.8, 6.5], dtype=np.float64)
  k_table = np.zeros((len(speeds), 4), dtype=np.float64)

  def sim_step(x: np.ndarray, u: float, v0: float) -> np.ndarray:
    mujoco.mj_resetData(model, d)
    lean, lean_rate, steer_q, steer_qd = x
    cr, sr = math.cos(0.5 * lean), math.sin(0.5 * lean)
    d.joint(f'{p}free').qpos[3:7] = [cr, sr, 0.0, 0.0]
    d.joint(f'{p}free').qvel[0] = v0
    d.joint(f'{p}free').qvel[3] = lean_rate
    d.joint(f'{p}rear_wheel_joint').qvel[0] = v0 / WHEEL_RADIUS
    d.joint(f'{p}front_wheel_joint').qvel[0] = v0 / WHEEL_RADIUS
    d.joint(f'{p}crank_joint').qvel[0] = v0 / (WHEEL_RADIUS * CHAIN_RATIO)
    d.joint(f'{p}steer_joint').qpos[0] = steer_q
    d.joint(f'{p}steer_joint').qvel[0] = steer_qd
    mujoco.mj_forward(model, d)
    d.actuator(f'{p}steer_servo').ctrl[0] = u
    mujoco.mj_step(model, d, nstep=SUBSTEPS)
    xmat = d.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
    fwd, up = xmat[:, 0], xmat[:, 2]
    yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    lh = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    lean_n = math.atan2(-float(np.dot(up, lh)), max(1e-4, float(up[2])))
    lr_n = float(d.sensor(f'{p}bike_gyro').data[0])
    return np.array([
        lean_n,
        lr_n,
        float(d.joint(f'{p}steer_joint').qpos[0]),
        float(d.joint(f'{p}steer_joint').qvel[0]),
    ])

  eps = 1e-4
  q_mat = np.diag([80.0, 3.0, 0.35, 0.015])
  for idx, v0 in enumerate(speeds):
    r_mat = np.array([[0.9 + 0.08 * float(v0)]])
    x0 = np.zeros(4)
    a_mat = np.zeros((4, 4))
    for i in range(4):
      dx = np.zeros(4)
      dx[i] = eps
      a_mat[:, i] = (sim_step(x0 + dx, 0.0, float(v0)) - sim_step(x0 - dx, 0.0, float(v0))) / (2.0 * eps)
    b_mat = ((sim_step(x0, eps, float(v0)) - sim_step(x0, -eps, float(v0))) / (2.0 * eps))[:, None]
    p_mat = linalg.solve_discrete_are(a_mat, b_mat, q_mat, r_mat)
    k_table[idx] = linalg.solve(r_mat + b_mat.T @ p_mat @ b_mat, b_mat.T @ p_mat @ a_mat)[0]
  return speeds, k_table


def supervisor_action_batch(
    obs_raw: np.ndarray,
    crank_qvel: np.ndarray,
    lqr_speeds: np.ndarray,
    lqr_k: np.ndarray,
) -> np.ndarray:
  """Computes normalized [-1, 1] supervisor actions for a batch of obs."""
  return policy_mod.lqr_supervisor_batch(obs_raw, crank_qvel, lqr_speeds, lqr_k)


class RunningMeanStd:
  """Online Welford mean/variance tracker for observation normalization."""

  def __init__(self, shape: int) -> None:
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = 1e-4

  def update(self, x: np.ndarray) -> None:
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
    self.mean = new_mean
    self.var = m2 / tot_count
    self.count = tot_count

  @property
  def std(self) -> np.ndarray:
    return np.sqrt(np.maximum(1e-3, self.var))


def ortho_init(
    rng: np.random.Generator, fan_in: int, fan_out: int, scale: float = 1.0
) -> np.ndarray:
  a = rng.standard_normal((max(fan_in, fan_out), min(fan_in, fan_out)))
  q, r = np.linalg.qr(a)
  q *= np.sign(np.diag(r))
  mat = q[:fan_in, :fan_out] if fan_in >= fan_out else q.T[:fan_in, :fan_out]
  return (scale * mat).astype(np.float64)


class AdamOptimizer:
  """Minimal Adam optimizer for a flat dict of NumPy parameter arrays."""

  def __init__(self, params: dict[str, np.ndarray], lr: float) -> None:
    self.lr = lr
    self.t = 0
    self.m = {k: np.zeros_like(v) for k, v in params.items()}
    self.v = {k: np.zeros_like(v) for k, v in params.items()}

  def step(
      self,
      params: dict[str, np.ndarray],
      grads: dict[str, np.ndarray],
      max_norm: float = 0.5,
  ) -> None:
    total_norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = min(1.0, max_norm / max(1e-8, total_norm))
    self.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr_t = self.lr * math.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t)
    for k, p in params.items():
      g = grads[k] * scale
      self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
      self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
      p -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + eps)


class ActorCritic:
  """Pure-NumPy Actor-Critic MLP with analytic backprop for BC & PPO."""

  def __init__(
      self,
      obs_dim: int,
      act_dim: int,
      hidden: int = 64,
      seed: int = 42,
      init_log_std: float = -1.7,
  ) -> None:
    rng = np.random.default_rng(seed)
    sq2 = math.sqrt(2.0)
    self.params: dict[str, np.ndarray] = {
        'aw1': ortho_init(rng, obs_dim, hidden, sq2),
        'ab1': np.zeros(hidden, dtype=np.float64),
        'aw2': ortho_init(rng, hidden, hidden, sq2),
        'ab2': np.zeros(hidden, dtype=np.float64),
        'aw3': ortho_init(rng, hidden, act_dim, 0.1),
        'ab3': np.zeros(act_dim, dtype=np.float64),
        'log_std': np.full(act_dim, init_log_std, dtype=np.float64),
        'cw1': ortho_init(rng, obs_dim, hidden, sq2),
        'cb1': np.zeros(hidden, dtype=np.float64),
        'cw2': ortho_init(rng, hidden, hidden, sq2),
        'cb2': np.zeros(hidden, dtype=np.float64),
        'cw3': ortho_init(rng, hidden, 1, 1.0),
        'cb3': np.zeros(1, dtype=np.float64),
    }

  def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    p = self.params
    ah1 = np.tanh(x @ p['aw1'] + p['ab1'])
    ah2 = np.tanh(ah1 @ p['aw2'] + p['ab2'])
    mu = np.tanh(ah2 @ p['aw3'] + p['ab3'])

    ch1 = np.tanh(x @ p['cw1'] + p['cb1'])
    ch2 = np.tanh(ch1 @ p['cw2'] + p['cb2'])
    val = (ch2 @ p['cw3'] + p['cb3']).squeeze(-1)

    std = np.exp(np.clip(p['log_std'], -2.8, -0.5))
    cache = (x, ah1, ah2, mu, ch1, ch2)
    return mu, std, val, cache

  def bc_gradients(
      self, x: np.ndarray, target_act: np.ndarray
  ) -> tuple[dict[str, np.ndarray], float]:
    p = self.params
    n = float(x.shape[0])
    mu, _, _, (x_c, ah1, ah2, mu_c, _, _) = self.forward(x)
    diff = mu_c - target_act
    loss = 0.5 * float(np.mean(np.sum(diff * diff, axis=-1)))
    d_mu = diff / n
    d_az3 = d_mu * (1.0 - mu_c * mu_c)
    g_aw3 = ah2.T @ d_az3
    g_ab3 = np.sum(d_az3, axis=0)
    d_az2 = (d_az3 @ p['aw3'].T) * (1.0 - ah2 * ah2)
    g_aw2 = ah1.T @ d_az2
    g_ab2 = np.sum(d_az2, axis=0)
    d_az1 = (d_az2 @ p['aw2'].T) * (1.0 - ah1 * ah1)
    g_aw1 = x_c.T @ d_az1
    g_ab1 = np.sum(d_az1, axis=0)
    return {
        'aw1': g_aw1,
        'ab1': g_ab1,
        'aw2': g_aw2,
        'ab2': g_ab2,
        'aw3': g_aw3,
        'ab3': g_ab3,
    }, loss

  def sample(
      self, rng: np.random.Generator, x: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, std, val, _ = self.forward(x)
    act = mu + std[None, :] * rng.standard_normal(mu.shape)
    logp = -0.5 * np.sum(
        ((act - mu) / std[None, :]) ** 2
        + 2.0 * np.log(std[None, :])
        + math.log(2.0 * math.pi),
        axis=-1,
    )
    return act, logp, val

  def ppo_gradients(
      self,
      x: np.ndarray,
      act: np.ndarray,
      old_logp: np.ndarray,
      adv: np.ndarray,
      ret: np.ndarray,
      clip_eps: float = 0.2,
      vf_coef: float = 0.5,
      ent_coef: float = 0.002,
      actor_active: bool = True,
  ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    p = self.params
    n = float(x.shape[0])
    mu, std, val, (x_c, ah1, ah2, mu_c, ch1, ch2) = self.forward(x)
    inv_std2 = 1.0 / (std * std)
    diff = act - mu_c
    log_std = np.log(std)
    logp = -0.5 * np.sum(
        (diff * diff) * inv_std2[None, :]
        + 2.0 * log_std[None, :]
        + math.log(2.0 * math.pi),
        axis=-1,
    )

    ratio = np.exp(np.clip(logp - old_logp, -10.0, 10.0))
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped_obj = ratio * adv
    clipped_obj = clipped * adv
    use_unclipped = (unclipped_obj <= clipped_obj).astype(np.float64)

    if actor_active:
      d_logp = -(1.0 / n) * use_unclipped * ratio * adv
      d_mu = d_logp[:, None] * diff * inv_std2[None, :]
      d_log_std = np.sum(
          d_logp[:, None] * (diff * diff * inv_std2[None, :] - 1.0), axis=0
      ) - ent_coef
      d_az3 = d_mu * (1.0 - mu_c * mu_c)
      g_aw3 = ah2.T @ d_az3
      g_ab3 = np.sum(d_az3, axis=0)
      d_az2 = (d_az3 @ p['aw3'].T) * (1.0 - ah2 * ah2)
      g_aw2 = ah1.T @ d_az2
      g_ab2 = np.sum(d_az2, axis=0)
      d_az1 = (d_az2 @ p['aw2'].T) * (1.0 - ah1 * ah1)
      g_aw1 = x_c.T @ d_az1
      g_ab1 = np.sum(d_az1, axis=0)
    else:
      g_aw1 = np.zeros_like(p['aw1'])
      g_ab1 = np.zeros_like(p['ab1'])
      g_aw2 = np.zeros_like(p['aw2'])
      g_ab2 = np.zeros_like(p['ab2'])
      g_aw3 = np.zeros_like(p['aw3'])
      g_ab3 = np.zeros_like(p['ab3'])
      d_log_std = np.zeros_like(p['log_std'])

    v_err = val - ret
    d_val_col = ((vf_coef / n) * v_err)[:, None]
    g_cw3 = ch2.T @ d_val_col
    g_cb3 = np.sum(d_val_col, axis=0)
    d_cz2 = (d_val_col @ p['cw3'].T) * (1.0 - ch2 * ch2)
    g_cw2 = ch1.T @ d_cz2
    g_cb2 = np.sum(d_cz2, axis=0)
    d_cz1 = (d_cz2 @ p['cw2'].T) * (1.0 - ch1 * ch1)
    g_cw1 = x_c.T @ d_cz1
    g_cb1 = np.sum(d_cz1, axis=0)

    grads = {
        'aw1': g_aw1,
        'ab1': g_ab1,
        'aw2': g_aw2,
        'ab2': g_ab2,
        'aw3': g_aw3,
        'ab3': g_ab3,
        'log_std': d_log_std,
        'cw1': g_cw1,
        'cb1': g_cb1,
        'cw2': g_cw2,
        'cb2': g_cb2,
        'cw3': g_cw3,
        'cb3': g_cb3,
    }
    stats = {
        'kl': float(np.mean(old_logp - logp)),
        'vloss': float(0.5 * np.mean(v_err * v_err)),
    }
    return grads, stats


class BatchBikeEnv:
  """Vectorized MuJoCo Batch environment across Figure-8, Cruise, and Circles."""

  def __init__(
      self, nsim: int = 64, nthread: int = 8, seed: int = 0
  ) -> None:
    self.nsim = nsim
    self.rng = np.random.default_rng(seed)
    self.model = build_training_model(with_posts=False)
    self.batch = batch.Batch(self.model, nsim=nsim, nthread=nthread)

    self.env_mode = np.zeros(nsim, dtype=np.int32)  # 0=fig8, 1=cruise, 2=circle_l, 3=circle_r
    self.circle_r = np.full(nsim, track.CIRCLE_RADIUS, dtype=np.float64)
    self.s_idx = np.zeros(nsim, dtype=np.int32)
    self.v_target = np.full(nsim, 3.4, dtype=np.float64)
    self.target_pitch = np.zeros(nsim, dtype=np.float64)
    self.wheelie_active = np.zeros(nsim, dtype=np.float64)
    self.no_hands_active = np.zeros(nsim, dtype=np.float64)
    self.carve_amp = np.zeros(nsim, dtype=np.float64)
    self.carve_phase = np.zeros(nsim, dtype=np.float64)
    self.ep_step = np.zeros(nsim, dtype=np.int32)
    self.prev_steer_cmd = np.zeros(nsim, dtype=np.float64)
    self.wing_mode = np.zeros(nsim, dtype=np.int32)
    self.max_ep_steps = 750

    self.reset_envs(np.arange(nsim))

  def reset_envs(self, ids: np.ndarray) -> None:
    if len(ids) == 0:
      return
    k = len(ids)
    self.batch.reset(ids=ids)

    # Mix: 45% Figure-8, 25% Straight Cruise, 15% Circle Left, 15% Circle Right
    modes = self.rng.choice([0, 1, 2, 3], size=k, p=[0.45, 0.25, 0.15, 0.15])
    self.env_mode[ids] = modes
    radii = self.rng.uniform(3.4, 6.2, size=k)
    # Bias half of circle envs to exact track.CIRCLE_RADIUS
    exact_r = self.rng.random(k) < 0.55
    radii = np.where(exact_r, track.CIRCLE_RADIUS, radii)
    self.circle_r[ids] = radii

    idx0 = self.rng.integers(0, track._N_FIG8, size=k)
    self.s_idx[ids] = idx0

    px0 = np.zeros(k, dtype=np.float64)
    py0 = np.zeros(k, dtype=np.float64)
    ryaw = np.zeros(k, dtype=np.float64)

    far_mask = self.rng.random(k) < 0.22
    lat_off = np.where(
        far_mask,
        self.rng.uniform(-5.5, 5.5, size=k),
        self.rng.uniform(-0.85, 0.85, size=k),
    )
    psi_off = np.where(
        far_mask,
        self.rng.uniform(-math.pi, math.pi, size=k),
        self.rng.uniform(-0.42, 0.42, size=k),
    )

    for j in range(k):
      md = int(modes[j])
      if md == 0:
        i0 = int(idx0[j])
        rx, ry = float(track.FIG8_PTS[i0, 0]), float(track.FIG8_PTS[i0, 1])
        yw = float(track.FIG8_YAWS[i0])
        px0[j] = rx - lat_off[j] * math.sin(yw)
        py0[j] = ry + lat_off[j] * math.cos(yw)
        ryaw[j] = yw
      elif md == 1:
        px0[j] = 0.0
        py0[j] = float(lat_off[j])
        ryaw[j] = 0.0
      else:
        # Either start near (0, 0) like SimPlugin or on the circle perimeter
        if self.rng.random() < 0.35:
          px0[j] = 0.0
          py0[j] = 0.0
          ryaw[j] = float(self.rng.uniform(-math.pi, math.pi))
        else:
          ang = float(self.rng.uniform(-math.pi, math.pi))
          r_c = float(radii[j])
          dir_c = 1.0 if md == 2 else -1.0
          r_spawn = max(1.0, r_c + float(lat_off[j]) * (-dir_c))
          px0[j] = r_spawn * math.cos(ang)
          py0[j] = r_spawn * math.sin(ang)
          ryaw[j] = ang + dir_c * 0.5 * math.pi

    lean0 = self.rng.uniform(-0.22, 0.22, size=k)
    v0 = self.rng.uniform(2.0, 6.0, size=k)
    vt = self.rng.uniform(2.0, 6.2, size=k)
    yaw0 = ryaw + psi_off

    cy, sy = np.cos(0.5 * yaw0), np.sin(0.5 * yaw0)
    cr, sr = np.cos(0.5 * lean0), np.sin(0.5 * lean0)

    free_qpos = self.batch.joint(f'{BIKE_PREFIX}free').qpos
    free_qvel = self.batch.joint(f'{BIKE_PREFIX}free').qvel
    for j, eid in enumerate(ids):
      free_qpos[eid, 0:3] = [px0[j], py0[j], 0.0]
      free_qpos[eid, 3:7] = [
          cy[j] * cr[j],
          cy[j] * sr[j],
          sy[j] * sr[j],
          sy[j] * cr[j],
      ]
      free_qvel[eid, 0:3] = [
          v0[j] * math.cos(yaw0[j]),
          v0[j] * math.sin(yaw0[j]),
          0.0,
      ]

    omega_w = v0 / WHEEL_RADIUS
    self.batch.joint(f'{BIKE_PREFIX}rear_wheel_joint').qvel[ids, 0] = omega_w
    self.batch.joint(f'{BIKE_PREFIX}front_wheel_joint').qvel[ids, 0] = omega_w
    self.batch.joint(f'{BIKE_PREFIX}crank_joint').qvel[ids, 0] = omega_w / CHAIN_RATIO

    self.batch.forward(ids=ids)

    self.v_target[ids] = vt
    self.target_pitch[ids] = 0.0
    self.wheelie_active[ids] = 0.0
    self.no_hands_active[ids] = 0.0
    self.carve_amp[ids] = 0.0
    self.carve_phase[ids] = self.rng.uniform(0, 2 * math.pi, size=k)
    self.ep_step[ids] = 0
    self.prev_steer_cmd[ids] = 0.0
    self.wing_mode[ids] = self.rng.integers(0, 3, size=k)

  def _get_state_and_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p = BIKE_PREFIX
    pos = self.batch.body(f'{p}pelican_bicycle').xpos
    xmat = self.batch.body(f'{p}pelican_bicycle').xmat.reshape(self.nsim, 3, 3)
    fwd = xmat[:, :, 0]
    up = xmat[:, :, 2]
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    left_h = np.column_stack([-np.sin(yaw), np.cos(yaw), np.zeros(self.nsim)])
    lean = np.arctan2(
        -np.einsum('ij,ij->i', up, left_h), np.maximum(1e-4, up[:, 2])
    )
    pitch = np.arcsin(np.clip(-xmat[:, 2, 0], -1.0, 1.0))

    vel = self.batch.sensor(f'{p}bike_vel').data
    gyro = self.batch.sensor(f'{p}bike_gyro').data
    v_fwd = vel[:, 0]
    v_lat = vel[:, 1]
    lean_rate = gyro[:, 0]
    pitch_rate = gyro[:, 1]
    yaw_rate = gyro[:, 2]

    steer_q = self.batch.joint(f'{p}steer_joint').qpos[:, 0]
    steer_qd = self.batch.joint(f'{p}steer_joint').qvel[:, 0]
    pr_q = self.batch.joint(f'{p}pelvis_roll').qpos[:, 0]
    pp_q = self.batch.joint(f'{p}pelvis_pitch').qpos[:, 0]
    front_z = self.batch.body(f'{p}front_wheel').xpos[:, 2] - WHEEL_RADIUS

    idx, f_elat, f_epsi, f_k0, f_k1, f_k2, f_k3, f_k4 = track.fig8_features_batch(
        pos[:, 0], pos[:, 1], yaw, self.s_idx
    )
    self.s_idx = idx

    # Cruise features (y=0, yaw=0)
    c_elat = pos[:, 1]
    c_epsi = track.wrap_pi(yaw)

    # Circle features (direction +1 for mode 2, -1 for mode 3)
    r_pos = np.maximum(1e-4, np.hypot(pos[:, 0], pos[:, 1]))
    rad_ang = np.arctan2(pos[:, 1], pos[:, 0])
    c_dir = np.where(self.env_mode == 2, 1.0, -1.0)
    tang_yaw = rad_ang + c_dir * 0.5 * math.pi
    cir_elat = np.where(c_dir >= 0.0, self.circle_r - r_pos, r_pos - self.circle_r)
    cir_epsi = track.wrap_pi(yaw - tang_yaw)
    cir_k = c_dir / self.circle_r

    is_fig8 = self.env_mode == 0
    is_cruise = self.env_mode == 1
    e_lat_raw = np.where(is_fig8, f_elat, np.where(is_cruise, c_elat, cir_elat))
    e_psi = np.where(is_fig8, f_epsi, np.where(is_cruise, c_epsi, cir_epsi))
    k0 = np.where(is_fig8, f_k0, np.where(is_cruise, 0.0, cir_k))
    k1 = np.where(is_fig8, f_k1, np.where(is_cruise, 0.0, cir_k))
    k2 = np.where(is_fig8, f_k2, np.where(is_cruise, 0.0, cir_k))
    k3 = np.where(is_fig8, f_k3, np.where(is_cruise, 0.0, cir_k))
    k4 = np.where(is_fig8, f_k4, np.where(is_cruise, 0.0, cir_k))

    t_sec = self.ep_step * 0.02
    carve_off = self.carve_amp * np.sin(
        2.0 * math.pi * 0.55 * t_sec + self.carve_phase
    )
    e_lat = e_lat_raw - carve_off

    max_k = np.maximum.reduce(
        [np.abs(k0), np.abs(k1), np.abs(k2), np.abs(k3), np.abs(k4)]
    )
    v_cap = np.where(max_k > 1e-4, np.sqrt(4.4 / np.maximum(1e-4, max_k)), 10.0)
    v_eff_target = np.maximum(1.8, np.minimum(self.v_target, v_cap))

    obs = policy_mod.build_obs_batch(
        lean=lean,
        lean_rate=lean_rate,
        pitch=pitch,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
        v_fwd=v_fwd,
        v_lat=v_lat,
        v_target=v_eff_target,
        steer_q=steer_q,
        steer_qd=steer_qd,
        pelvis_roll_q=pr_q,
        pelvis_pitch_q=pp_q,
        e_lat=e_lat,
        e_psi=e_psi,
        k0=k0,
        k1=k1,
        k2=k2,
        k3=k3,
        target_pitch=self.target_pitch,
        wheelie_active=self.wheelie_active,
        no_hands_active=self.no_hands_active,
    )
    info = {
        'lean': lean,
        'lean_rate': lean_rate,
        'pitch': pitch,
        'pitch_rate': pitch_rate,
        'up_z': up[:, 2],
        'v_fwd': v_fwd,
        'v_eff_target': v_eff_target,
        'steer_qd': steer_qd,
        'e_lat': e_lat,
        'e_lat_raw': e_lat_raw,
        'e_psi': e_psi,
        'k0': k0,
        'front_z': front_z,
        'crank_qvel': self.batch.joint(f'{p}crank_joint').qvel[:, 0],
    }
    return obs, info

  def step(
      self,
      act_norm: np.ndarray,
      *,
      enable_disturbances: bool = True,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    p = BIKE_PREFIX
    a_clipped = np.clip(act_norm, -1.0, 1.0)
    a_phys = (
        a_clipped * policy_mod.ACT_SCALE[None, :]
        + policy_mod.ACT_BIAS[None, :]
    )
    steer_cmd = a_phys[:, 0]
    drive_cmd = a_phys[:, 1]
    pr_cmd = a_phys[:, 2]
    pp_cmd = a_phys[:, 3]

    if enable_disturbances:
      push_mask = (self.rng.random(self.nsim) < 0.008) & (self.ep_step > 25)
      if np.any(push_mask):
        pids = np.where(push_mask)[0]
        xmat = self.batch.body(f'{p}pelican_bicycle').xmat.reshape(
            self.nsim, 3, 3
        )
        left_vec = xmat[pids, :, 1]
        sign = self.rng.choice([-1.0, 1.0], size=len(pids))
        mag = self.rng.uniform(0.30, 0.65, size=len(pids))
        fqv = self.batch.joint(f'{p}free').qvel
        fqv[pids, 0] += sign * mag * left_vec[:, 0]
        fqv[pids, 1] += sign * mag * left_vec[:, 1]
        fqv[pids, 3] -= sign * mag * 0.75

      # Occasional airborne hop landing impulse (~0.4% chance)
      hop_mask = (self.rng.random(self.nsim) < 0.004) & (self.ep_step > 30) & (self.wheelie_active < 0.5)
      if np.any(hop_mask):
        hids = np.where(hop_mask)[0]
        self.batch.joint(f'{p}free').qvel[hids, 2] += self.rng.uniform(0.9, 1.4, size=len(hids))

      trans_mask = (self.ep_step > 20) & ((self.ep_step % 75) == 0)
      if np.any(trans_mask):
        xm_b = self.batch.body(f'{p}pelican_bicycle').xmat.reshape(self.nsim, 3, 3)
        yaw_b = np.arctan2(xm_b[:, 1, 0], xm_b[:, 0, 0])
        lh_b = np.column_stack([-np.sin(yaw_b), np.cos(yaw_b), np.zeros(self.nsim)])
        lean_cur = np.arctan2(
            -np.einsum('ij,ij->i', xm_b[:, :, 2], lh_b),
            np.maximum(1e-4, xm_b[:, 2, 2]),
        )
        for eid in np.where(trans_mask)[0]:
          roll = float(self.rng.random())
          straight_enough = (self.env_mode[eid] == 1) or (abs(float(lean_cur[eid])) < 0.16)
          if roll < 0.25 and straight_enough:
            # Wheelie phase only on low-lean/straight segments
            self.wheelie_active[eid] = 1.0
            self.target_pitch[eid] = -0.24
            self.v_target[eid] = float(self.rng.uniform(2.6, 3.8))
            self.carve_amp[eid] = 0.0
            self.no_hands_active[eid] = 0.0
          elif roll < 0.50:
            # Turbo boost / high-speed sprint phase
            self.wheelie_active[eid] = 0.0
            self.target_pitch[eid] = 0.0
            self.v_target[eid] = float(self.rng.uniform(5.2, 6.2))
            self.carve_amp[eid] = 0.0
            self.no_hands_active[eid] = 0.0
          elif roll < 0.68:
            # S-Carve phase
            self.wheelie_active[eid] = 0.0
            self.target_pitch[eid] = 0.0
            self.v_target[eid] = float(self.rng.uniform(2.6, 4.2))
            self.carve_amp[eid] = float(self.rng.uniform(0.35, 0.55))
            self.no_hands_active[eid] = 0.0
          elif roll < 0.80:
            # No-Hands mode
            self.wheelie_active[eid] = 0.0
            self.target_pitch[eid] = 0.0
            self.v_target[eid] = float(self.rng.uniform(2.6, 4.2))
            self.carve_amp[eid] = 0.0
            self.no_hands_active[eid] = 1.0
          else:
            self.wheelie_active[eid] = 0.0
            self.target_pitch[eid] = 0.0
            self.v_target[eid] = float(self.rng.uniform(2.0, 5.2))
            self.carve_amp[eid] = 0.0
            self.no_hands_active[eid] = 0.0

    self.batch.actuator(f'{p}steer_servo').ctrl[:, 0] = steer_cmd
    self.batch.actuator(f'{p}drive_motor').ctrl[:, 0] = drive_cmd
    # Automatic gentle corner-entry front brake when overspeeding
    v_fwd_cur = self.batch.sensor(f'{p}bike_vel').data[:, 0]
    brake_cmd = np.where(
        (self.wheelie_active < 0.5) & (v_fwd_cur > self.v_target + 0.4),
        np.clip(22.0 * (self.v_target - v_fwd_cur), -45.0, 0.0),
        0.0,
    )
    self.batch.actuator(f'{p}front_brake').ctrl[:, 0] = brake_cmd
    self.batch.actuator(f'{p}pelvis_roll_servo').ctrl[:, 0] = pr_cmd
    self.batch.actuator(f'{p}pelvis_pitch_servo').ctrl[:, 0] = pp_cmd

    t_arr = self.ep_step * 0.02
    flap_mask = self.wing_mode == 2
    spread_mask = (self.wing_mode == 1) | (self.no_hands_active > 0.5)
    for side in ('l', 'r'):
      self.batch.actuator(f'{p}wing_{side}_yaw_servo').ctrl[:, 0] = np.where(
          flap_mask | spread_mask, 1.05, 0.0
      )
      self.batch.actuator(f'{p}wing_{side}_flap_servo').ctrl[:, 0] = np.where(
          flap_mask,
          0.18 + 0.65 * np.sin(7.2 * t_arr),
          np.where(spread_mask, 0.25, 0.0),
      )
      self.batch.actuator(f'{p}wing_{side}_pitch_servo').ctrl[:, 0] = 0.0
      self.batch.actuator(f'{p}wing_{side}_elbow_servo').ctrl[:, 0] = np.where(
          flap_mask, -0.45 * np.cos(7.2 * t_arr), 0.0
      )

    # Keep pedals & ankles horizontal (feedforward crank midpoint for speed!)
    cq = self.batch.joint(f'{p}crank_joint').qpos[:, 0]
    cqd = self.batch.joint(f'{p}crank_joint').qvel[:, 0]
    cq_mid = cq + 0.01 * cqd
    pp = self.batch.joint(f'{p}pelvis_pitch').qpos[:, 0]
    self.batch.actuator(f'{p}pedal_l_servo').ctrl[:, 0] = -cq_mid
    self.batch.actuator(f'{p}pedal_r_servo').ctrl[:, 0] = -cq_mid
    self.batch.actuator(f'{p}ankle_l_servo').ctrl[:, 0] = -(
        self.batch.joint(f'{p}hip_l_pitch').qpos[:, 0]
        + self.batch.joint(f'{p}knee_l_pitch').qpos[:, 0]
        + pp
    )
    self.batch.actuator(f'{p}ankle_r_servo').ctrl[:, 0] = -(
        self.batch.joint(f'{p}hip_r_pitch').qpos[:, 0]
        + self.batch.joint(f'{p}knee_r_pitch').qpos[:, 0]
        + pp
    )

    self.batch.step(nstep=SUBSTEPS)
    self.ep_step += 1
    next_obs, info = self._get_state_and_obs()

    e_lat = info['e_lat']
    e_psi = info['e_psi']
    v_fwd = info['v_fwd']
    v_tgt = info['v_eff_target']
    lean = info['lean']
    pitch = info['pitch']
    front_z = info['front_z']

    phi_coord = np.clip(
        -np.arctan((v_fwd * v_fwd * info['k0']) / 9.81), -0.55, 0.55
    )
    r_upright = (
        1.2 * np.exp(-4.0 * (lean - phi_coord) ** 2)
        + 0.6 * info['up_z']
        - 0.06 * info['lean_rate'] ** 2
    )
    # Vector-field alignment rewards heading smoothly toward road even when far off track
    l_look = np.maximum(2.8, 1.05 * np.maximum(1.6, v_fwd))
    e_psi_vf = track.wrap_pi(e_psi - np.arctan2(-e_lat, l_look))
    r_track = (
        1.8 * np.exp(-2.8 * e_lat * e_lat)
        + 1.0 * np.exp(-3.0 * e_psi * e_psi)
        + 0.8 * np.exp(-2.5 * e_psi_vf * e_psi_vf)
        - 0.35 * np.minimum(6.0, np.abs(e_lat))
    )
    r_speed = 1.2 * np.exp(-1.5 * (v_fwd - v_tgt) ** 2) * (v_fwd > 0.8)
    r_wheelie = np.where(
        self.wheelie_active > 0.5,
        2.2 * np.exp(-10.0 * (pitch - self.target_pitch) ** 2)
        + 1.5 * np.clip(front_z / 0.25, 0.0, 1.4),
        -0.3 * (pitch * pitch),
    )
    r_smooth = (
        -0.05 * (steer_cmd - self.prev_steer_cmd) ** 2
        - 0.015 * info['steer_qd'] ** 2
    )
    self.prev_steer_cmd = steer_cmd.copy()

    reward = 0.15 * (r_upright + r_track + r_speed + r_wheelie + r_smooth)
    fell = (
        (np.abs(lean) > 0.96)
        | (info['up_z'] < 0.48)
        | (np.abs(info['e_lat_raw']) > 8.5)
    )
    timeout = self.ep_step >= self.max_ep_steps
    done = fell | timeout
    reward = np.where(fell, reward - 4.0, reward)

    metrics = {
        'mean_err': float(np.mean(np.abs(info['e_lat_raw']))),
        'fell_frac': float(np.mean(fell)),
        'mean_speed': float(np.mean(v_fwd)),
    }

    if np.any(done):
      self.reset_envs(np.where(done)[0])
      next_obs, _ = self._get_state_and_obs()

    return next_obs, reward, done.astype(np.float64), metrics


def evaluate_policy_suite(pol: policy_mod.PelicanPolicy) -> dict[str, float]:
  """Runs deterministic full-scene suite across Figure-8 (with posts), Cruise, Circle, Off-track."""
  m_posts = build_training_model(with_posts=True)
  m_clean = build_training_model(with_posts=False)
  p = BIKE_PREFIX

  def run_one(
      m: mujoco.MjModel,
      mode: str,
      target_speed: float,
      seconds: float,
      nudge_times: tuple[float, ...] = (),
      wheelie_window: tuple[float, float] | None = None,
      start_pose: tuple[float, float, float] | None = None,
  ) -> dict[str, float]:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if start_pose is not None:
      sx, sy, yaw0 = start_pose
    else:
      sx, sy = 0.0, 0.0
      yaw0 = track.FIG8_START_YAW if mode == 'figure8' else 0.0
    d.joint(f'{p}free').qpos[0:3] = [sx, sy, 0.0]
    d.joint(f'{p}free').qpos[3:7] = [
        math.cos(0.5 * yaw0),
        0.0,
        0.0,
        math.sin(0.5 * yaw0),
    ]
    d.joint(f'{p}free').qvel[0:3] = [
        target_speed * math.cos(yaw0),
        target_speed * math.sin(yaw0),
        0.0,
    ]
    omega_w = target_speed / WHEEL_RADIUS
    d.joint(f'{p}rear_wheel_joint').qvel[0] = omega_w
    d.joint(f'{p}front_wheel_joint').qvel[0] = omega_w
    d.joint(f'{p}crank_joint').qvel[0] = omega_w / CHAIN_RATIO
    mujoco.mj_forward(m, d)

    s_idx = 0 if (mode == 'figure8' and start_pose is None) else None
    n_steps = int(seconds / 0.02)
    nudge_steps = {round(nt / 0.02) for nt in nudge_times}
    errs = []
    max_fz = 0.0
    max_lean = 0.0
    dist_m = 0.0
    prev_xy = np.array([sx, sy])
    fell = False

    for step in range(n_steps):
      t = step * 0.02
      if step in nudge_steps:
        xmat = d.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
        lv = xmat[:, 1]
        fqv = d.joint(f'{p}free').qvel
        fqv[0] += 0.55 * float(lv[0])
        fqv[1] += 0.55 * float(lv[1])
        fqv[3] -= 0.65

      pos = d.body(f'{p}pelican_bicycle').xpos
      dist_m += float(math.hypot(float(pos[0]) - prev_xy[0], float(pos[1]) - prev_xy[1]))
      prev_xy = np.array([float(pos[0]), float(pos[1])])

      xmat = d.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
      fwd, up = xmat[:, 0], xmat[:, 2]
      yaw = math.atan2(float(fwd[1]), float(fwd[0]))
      lh = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
      lean = math.atan2(-float(np.dot(up, lh)), max(1e-4, float(up[2])))
      max_lean = max(max_lean, abs(math.degrees(lean)))
      pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))
      vel = d.sensor(f'{p}bike_vel').data
      gyro = d.sensor(f'{p}bike_gyro').data
      v_fwd = float(vel[0])

      if mode == 'figure8':
        s_idx, e_lat, e_psi, k0, k1, k2, k3, k4 = track.fig8_features(
            float(pos[0]), float(pos[1]), yaw, prev_idx=s_idx
        )
      elif mode == 'circle_left':
        e_lat, e_psi, k0, k1, k2, k3 = track.circle_features(
            float(pos[0]), float(pos[1]), yaw, direction=1.0
        )
        k4 = k3
      else:
        e_lat, e_psi, k0, k1, k2, k3 = track.cruise_features(
            float(pos[0]), float(pos[1]), yaw, 0.0, 0.0, 0.0
        )
        k4 = k3
      v_tgt = track.safe_corner_speed(target_speed, k0, k1, k2, k3, k4)

      in_w = wheelie_window is not None and (wheelie_window[0] <= t <= wheelie_window[1])
      if in_w:
        ramp = min(1.0, (t - wheelie_window[0]) / 0.16, (wheelie_window[1] - t) / 0.25)
        tgt_p = -0.24 * max(0.0, ramp)
      else:
        tgt_p = 0.0

      obs = policy_mod.build_obs_single(
          lean=lean,
          lean_rate=float(gyro[0]),
          pitch=pitch,
          pitch_rate=float(gyro[1]),
          yaw_rate=float(gyro[2]),
          v_fwd=v_fwd,
          v_lat=float(vel[1]),
          v_target=v_tgt,
          steer_q=float(d.joint(f'{p}steer_joint').qpos[0]),
          steer_qd=float(d.joint(f'{p}steer_joint').qvel[0]),
          pelvis_roll_q=float(d.joint(f'{p}pelvis_roll').qpos[0]),
          pelvis_pitch_q=float(d.joint(f'{p}pelvis_pitch').qpos[0]),
          e_lat=e_lat,
          e_psi=e_psi,
          k0=k0,
          k1=k1,
          k2=k2,
          k3=k3,
          target_pitch=tgt_p,
          wheelie_active=1.0 if in_w else 0.0,
      )
      act = pol.act(obs)
      d.actuator(f'{p}steer_servo').ctrl[0] = float(act[0])
      d.actuator(f'{p}drive_motor').ctrl[0] = float(act[1])
      d.actuator(f'{p}front_brake').ctrl[0] = (
          max(-45.0, min(0.0, 22.0 * (v_tgt - v_fwd)))
          if (not in_w and v_fwd > v_tgt + 0.4)
          else 0.0
      )
      d.actuator(f'{p}pelvis_roll_servo').ctrl[0] = float(act[2])
      d.actuator(f'{p}pelvis_pitch_servo').ctrl[0] = float(act[3])

      cq = float(d.joint(f'{p}crank_joint').qpos[0])
      cqd = float(d.joint(f'{p}crank_joint').qvel[0])
      cq_mid = cq + 0.01 * cqd
      pp = float(d.joint(f'{p}pelvis_pitch').qpos[0])
      d.actuator(f'{p}pedal_l_servo').ctrl[0] = -cq_mid
      d.actuator(f'{p}pedal_r_servo').ctrl[0] = -cq_mid
      d.actuator(f'{p}ankle_l_servo').ctrl[0] = -(
          float(d.joint(f'{p}hip_l_pitch').qpos[0])
          + float(d.joint(f'{p}knee_l_pitch').qpos[0])
          + pp
      )
      d.actuator(f'{p}ankle_r_servo').ctrl[0] = -(
          float(d.joint(f'{p}hip_r_pitch').qpos[0])
          + float(d.joint(f'{p}knee_r_pitch').qpos[0])
          + pp
      )
      mujoco.mj_step(m, d, nstep=SUBSTEPS)

      errs.append(abs(e_lat))
      if in_w:
        max_fz = max(max_fz, float(d.body(f'{p}front_wheel').xpos[2]) - WHEEL_RADIUS)
      if abs(lean) > 0.98 or float(up[2]) < 0.45:
        fell = True
        break

    return {
        'surv_s': float(d.time),
        'fell': 1.0 if fell else 0.0,
        'mean_err_m': float(np.mean(errs)) if errs else 0.0,
        'max_err_m': float(np.max(errs)) if errs else 0.0,
        'max_lean_deg': max_lean,
        'dist_m': dist_m,
        'wheelie_cm': max_fz * 100.0,
    }

  c34 = run_one(m_posts, 'figure8', 3.4, 30.0)
  p34 = run_one(m_posts, 'figure8', 3.4, 30.0, nudge_times=(10.0, 20.0))
  b58 = run_one(m_posts, 'figure8', 5.8, 30.0)
  w34 = run_one(m_posts, 'figure8', 3.4, 12.0, wheelie_window=(0.10, 0.78))
  s58 = run_one(m_clean, 'cruise', 5.8, 15.0)
  cl4 = run_one(m_clean, 'circle_left', 4.5, 15.0)
  off = run_one(m_clean, 'figure8', 3.4, 15.0, start_pose=(0.0, 8.0, math.pi))
  return {
      'cruise_surv': c34['surv_s'],
      'cruise_err': c34['mean_err_m'],
      'cruise_max': c34['max_err_m'],
      'cruise_dist': c34['dist_m'],
      'push_surv': p34['surv_s'],
      'push_err': p34['mean_err_m'],
      'boost_surv': b58['surv_s'],
      'boost_err': b58['mean_err_m'],
      'wheelie_surv': w34['surv_s'],
      'wheelie_cm': w34['wheelie_cm'],
      'straight58_lean': s58['max_lean_deg'],
      'straight58_err': s58['mean_err_m'],
      'circle_err': cl4['mean_err_m'],
      'offtrack_err': off['mean_err_m'],
      'total_fell': c34['fell'] + p34['fell'] + b58['fell'] + w34['fell'] + s58['fell'] + cl4['fell'] + off['fell'],
  }


def train(
    bc_epochs: int = 80,
    ppo_iters: int = 40,
    nsim: int = 64,
    rollout_len: int = 128,
    seed: int = 42,
    out_path: str = policy_mod.DEFAULT_POLICY_PATH,
) -> None:
  t0 = time.perf_counter()
  rng = np.random.default_rng(seed)
  env = BatchBikeEnv(nsim=nsim, nthread=8, seed=seed)
  lqr_speeds, lqr_k = compute_inner_lqr_table(env.model)
  print(f'Computed LQR table in {time.perf_counter() - t0:.2f}s', flush=True)

  ac = ActorCritic(
      policy_mod.OBS_DIM,
      policy_mod.ACT_DIM,
      hidden=96,
      seed=seed,
      init_log_std=-1.65,
  )
  actor_keys = ('aw1', 'ab1', 'aw2', 'ab2', 'aw3', 'ab3')
  ppo_actor_keys = actor_keys + ('log_std',)
  critic_keys = ('cw1', 'cb1', 'cw2', 'cb2', 'cw3', 'cb3')
  actor_params = {k: ac.params[k] for k in ppo_actor_keys}
  critic_params = {k: ac.params[k] for k in critic_keys}
  rms = RunningMeanStd(policy_mod.OBS_DIM)

  # --- Phase 1: DAgger + Behavioral Cloning Warmstart ---
  bc_obs_list = []
  bc_act_list = []
  opt_bc = AdamOptimizer({k: ac.params[k] for k in actor_keys}, lr=3e-3)

  for dagger_round in range(3):
    env.reset_envs(np.arange(nsim))
    obs_raw, info = env._get_state_and_obs()
    mix_student = 0.0 if dagger_round == 0 else (0.5 if dagger_round == 1 else 0.85)
    n_steps = 220 if dagger_round == 0 else 180
    for step in range(n_steps):
      rms.update(obs_raw)
      expert_act = supervisor_action_batch(obs_raw, info['crank_qvel'], lqr_speeds, lqr_k)
      bc_obs_list.append(obs_raw.copy())
      bc_act_list.append(expert_act.copy())

      if mix_student > 0.0:
        obs_norm = np.clip((obs_raw - rms.mean) / rms.std, -8.0, 8.0)
        stud_mu, _, _, _ = ac.forward(obs_norm)
        act_exec = (1.0 - mix_student) * expert_act + mix_student * stud_mu
      else:
        act_exec = expert_act
      act_exec = np.clip(
          act_exec + rng.normal(0.0, 0.06, size=act_exec.shape), -1.0, 1.0
      )
      obs_raw, _, _, _ = env.step(act_exec, enable_disturbances=True)
      _, info = env._get_state_and_obs()

    all_obs = np.concatenate(bc_obs_list, axis=0)
    all_act = np.concatenate(bc_act_list, axis=0)
    all_obs_norm = np.clip((all_obs - rms.mean) / rms.std, -8.0, 8.0)

    n_data = len(all_obs_norm)
    epochs_here = bc_epochs // 2 if dagger_round < 2 else bc_epochs
    for ep in range(epochs_here):
      perm = rng.permutation(n_data)
      losses = []
      for s in range(0, n_data, 1024):
        mb = perm[s : s + 1024]
        grads, loss = ac.bc_gradients(all_obs_norm[mb], all_act[mb])
        opt_bc.step({k: ac.params[k] for k in actor_keys}, grads, max_norm=1.0)
        losses.append(loss)
    print(
        f'[Phase 1 DAgger round {dagger_round + 1}/3 | samples={n_data:,} | {time.perf_counter() - t0:.1f}s] '
        f'bc_loss={np.mean(losses):.5f}',
        flush=True,
    )

  def save_if_best(tag: str, ev: dict[str, float], best_s: float, it_num: float) -> float:
    # Penalize falling OR stalling (requires >= 65m on 30s cruise!)
    stall_pen = 30.0 if ev['cruise_dist'] < 65.0 else 0.0
    score = (
        ev['cruise_surv']
        + ev['push_surv']
        + ev['boost_surv']
        + 1.5 * ev['wheelie_surv']
        + 0.35 * min(35.0, ev['wheelie_cm'])
        - 10.0 * ev['cruise_err']
        - 6.0 * ev['push_err']
        - 6.0 * ev['boost_err']
        - 0.25 * ev['straight58_lean']
        - 4.0 * ev['circle_err']
        - 1.5 * ev['offtrack_err']
        - 25.0 * ev['total_fell']
        - stall_pen
    )
    print(
        f'[{tag} | {time.perf_counter() - t0:5.1f}s] '
        f'cruise={ev["cruise_surv"]:4.1f}s({ev["cruise_err"]:.3f}m) '
        f'boost={ev["boost_surv"]:4.1f}s({ev["boost_err"]:.3f}m) '
        f'str58_lean={ev["straight58_lean"]:4.1f}d '
        f'cir={ev["circle_err"]:.2f}m off={ev["offtrack_err"]:.2f}m '
        f'wheelie={ev["wheelie_cm"]:4.1f}cm '
        f'fell={ev["total_fell"]:.0f} score={score:+.1f}',
        flush=True,
    )
    if score >= best_s:
      tmp_path = out_path + '.tmp.npz'
      np.savez(
          tmp_path,
          obs_mean=rms.mean,
          obs_std=rms.std,
          w1=ac.params['aw1'],
          b1=ac.params['ab1'],
          w2=ac.params['aw2'],
          b2=ac.params['ab2'],
          w3=ac.params['aw3'],
          b3=ac.params['ab3'],
          lqr_speeds=lqr_speeds,
          lqr_k=lqr_k,
          iter=np.array(it_num, dtype=np.float64),
          eval_err_m=np.array(ev['cruise_err'], dtype=np.float64),
          eval_surv_s=np.array(ev['cruise_surv'], dtype=np.float64),
          eval_wheelie_deg=np.array(ev['wheelie_cm'], dtype=np.float64),
      )
      os.replace(tmp_path, out_path)
      return score
    return best_s

  def make_pol() -> policy_mod.PelicanPolicy:
    return policy_mod.PelicanPolicy(
        obs_mean=rms.mean,
        obs_std=rms.std,
        w1=ac.params['aw1'],
        b1=ac.params['ab1'],
        w2=ac.params['aw2'],
        b2=ac.params['ab2'],
        w3=ac.params['aw3'],
        b3=ac.params['ab3'],
        lqr_speeds=lqr_speeds,
        lqr_k=lqr_k,
    )

  best_score = save_if_best('Phase 1 BC-MLP', evaluate_policy_suite(make_pol()), -1e9, 0.0)

  # --- Phase 2: PPO RL Fine-Tuning (Critic warmup + full Actor-Critic PPO) ---
  opt_actor = AdamOptimizer(actor_params, lr=2.5e-4)
  opt_critic = AdamOptimizer(critic_params, lr=1.5e-3)
  env.reset_envs(np.arange(nsim))
  obs_raw, _ = env._get_state_and_obs()

  total_ppo = ppo_iters + 5  # first 5 iters: critic warmup only!
  for it in range(1, total_ppo + 1):
    actor_active = it > 5
    if actor_active:
      frac = 1.0 - (it - 6) / max(1, ppo_iters)
      opt_actor.lr = 2.5e-4 * (0.2 + 0.8 * frac)

    obs_buf = np.zeros((rollout_len, nsim, policy_mod.OBS_DIM), dtype=np.float64)
    act_buf = np.zeros((rollout_len, nsim, policy_mod.ACT_DIM), dtype=np.float64)
    logp_buf = np.zeros((rollout_len, nsim), dtype=np.float64)
    val_buf = np.zeros((rollout_len, nsim), dtype=np.float64)
    rew_buf = np.zeros((rollout_len, nsim), dtype=np.float64)
    done_buf = np.zeros((rollout_len, nsim), dtype=np.float64)

    falls = 0.0
    errs = []
    for t in range(rollout_len):
      obs_norm = np.clip((obs_raw - rms.mean) / rms.std, -8.0, 8.0)
      act, logp, val = ac.sample(rng, obs_norm)
      obs_buf[t] = obs_norm
      act_buf[t] = act
      logp_buf[t] = logp
      val_buf[t] = val
      obs_raw, rew, done, m_info = env.step(act, enable_disturbances=True)
      rew_buf[t] = rew
      done_buf[t] = done
      falls += m_info['fell_frac'] * nsim
      errs.append(m_info['mean_err'])

    last_norm = np.clip((obs_raw - rms.mean) / rms.std, -8.0, 8.0)
    _, _, last_val, _ = ac.forward(last_norm)

    gamma, gae_lam = 0.99, 0.95
    adv_buf = np.zeros_like(rew_buf)
    last_gae = np.zeros(nsim, dtype=np.float64)
    for t in reversed(range(rollout_len)):
      nt = 1.0 - done_buf[t]
      nv = last_val if t == rollout_len - 1 else val_buf[t + 1]
      delta = rew_buf[t] + gamma * nv * nt - val_buf[t]
      last_gae = delta + gamma * gae_lam * nt * last_gae
      adv_buf[t] = last_gae
    ret_buf = adv_buf + val_buf

    n_total = rollout_len * nsim
    b_obs = obs_buf.reshape(n_total, policy_mod.OBS_DIM)
    b_act = act_buf.reshape(n_total, policy_mod.ACT_DIM)
    b_logp = logp_buf.reshape(n_total)
    b_adv = (adv_buf.reshape(n_total) - np.mean(adv_buf)) / (np.std(adv_buf) + 1e-8)
    b_ret = ret_buf.reshape(n_total)

    for _ in range(4):
      perm = rng.permutation(n_total)
      kl_sum = 0.0
      n_mb = 0
      for s in range(0, n_total, 1024):
        mb = perm[s : s + 1024]
        grads, stats = ac.ppo_gradients(
            b_obs[mb], b_act[mb], b_logp[mb], b_adv[mb], b_ret[mb],
            actor_active=actor_active,
        )
        if actor_active:
          opt_actor.step(actor_params, {k: grads[k] for k in ppo_actor_keys}, max_norm=0.5)
        opt_critic.step(critic_params, {k: grads[k] for k in critic_keys}, max_norm=2.0)
        kl_sum += stats['kl']
        n_mb += 1
      if actor_active and (kl_sum / max(1, n_mb)) > 0.02:
        break

    if it == 5 or (actor_active and ((it - 5) % 5 == 0 or it == total_ppo)):
      ppo_step = it - 5
      ev = evaluate_policy_suite(make_pol())
      tag = 'Critic Warmup' if not actor_active else f'PPO iter {ppo_step:2d}/{ppo_iters}'
      if actor_active:
        best_score = save_if_best(tag, ev, best_score, float(ppo_step))
      else:
        print(f'[{tag}] train_rew={np.mean(rew_buf):+.3f} train_err={np.mean(errs):.3f}m falls={falls:.0f} vloss={stats["vloss"]:.3f}', flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--bc_epochs', type=int, default=70)
  parser.add_argument('--ppo_iters', type=int, default=35)
  parser.add_argument('--nsim', type=int, default=64)
  parser.add_argument('--rollout_len', type=int, default=128)
  args = parser.parse_args()
  train(
      bc_epochs=args.bc_epochs,
      ppo_iters=args.ppo_iters,
      nsim=args.nsim,
      rollout_len=args.rollout_len,
  )
