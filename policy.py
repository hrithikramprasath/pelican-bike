"""NumPy MLP policy inference for the Pelican Bicycle controller.

Stores actor weights, critic weights (optional at runtime), and running
observation normalization statistics in ``trained_policy.npz``.
"""

from __future__ import annotations

import os
import numpy as np

OBS_DIM = 22
ACT_DIM = 4

ACT_SCALE = np.array([0.72, 300.0, 0.36, 0.37], dtype=np.float64)
ACT_BIAS = np.array([0.0, 0.0, 0.0, -0.01], dtype=np.float64)

DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'trained_policy.npz'
)


def build_obs_single(
    lean: float,
    lean_rate: float,
    pitch: float,
    pitch_rate: float,
    yaw_rate: float,
    v_fwd: float,
    v_lat: float,
    v_target: float,
    steer_q: float,
    steer_qd: float,
    pelvis_roll_q: float,
    pelvis_pitch_q: float,
    e_lat: float,
    e_psi: float,
    k0: float,
    k1: float,
    k2: float,
    k3: float,
    target_pitch: float = 0.0,
    wheelie_active: float = 0.0,
    no_hands_active: float = 0.0,
) -> np.ndarray:
  """Constructs a single (22,) observation vector."""
  return np.array(
      [
          lean,
          lean_rate,
          pitch,
          pitch_rate,
          yaw_rate,
          v_fwd,
          v_lat,
          v_target,
          v_target - v_fwd,
          steer_q,
          steer_qd,
          pelvis_roll_q,
          pelvis_pitch_q,
          max(-6.0, min(6.0, e_lat)),
          e_psi,
          k0,
          k1,
          k2,
          k3,
          target_pitch,
          wheelie_active,
          no_hands_active,
      ],
      dtype=np.float64,
  )


def build_obs_batch(
    lean: np.ndarray,
    lean_rate: np.ndarray,
    pitch: np.ndarray,
    pitch_rate: np.ndarray,
    yaw_rate: np.ndarray,
    v_fwd: np.ndarray,
    v_lat: np.ndarray,
    v_target: np.ndarray,
    steer_q: np.ndarray,
    steer_qd: np.ndarray,
    pelvis_roll_q: np.ndarray,
    pelvis_pitch_q: np.ndarray,
    e_lat: np.ndarray,
    e_psi: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    target_pitch: np.ndarray,
    wheelie_active: np.ndarray,
    no_hands_active: np.ndarray,
) -> np.ndarray:
  """Constructs vectorized (N, 22) observation matrix."""
  return np.column_stack([
      lean,
      lean_rate,
      pitch,
      pitch_rate,
      yaw_rate,
      v_fwd,
      v_lat,
      v_target,
      v_target - v_fwd,
      steer_q,
      steer_qd,
      pelvis_roll_q,
      pelvis_pitch_q,
      np.clip(e_lat, -6.0, 6.0),
      e_psi,
      k0,
      k1,
      k2,
      k3,
      target_pitch,
      wheelie_active,
      no_hands_active,
  ])


WHEEL_RADIUS = 0.330
WHEELBASE = 1.090
HEAD_ANGLE_RAD = 0.3490658503988659  # math.radians(20.0)
CHAIN_RATIO = 2.0


def lqr_supervisor_batch(
    obs_raw: np.ndarray,
    crank_qvel: np.ndarray,
    lqr_speeds: np.ndarray,
    lqr_k: np.ndarray,
) -> np.ndarray:
  """Computes normalized [-1, 1] speed-invariant LQR + vector-field actions."""
  lean = obs_raw[:, 0]
  lean_rate = obs_raw[:, 1]
  pitch = obs_raw[:, 2]
  pitch_rate = obs_raw[:, 3]
  v_fwd = obs_raw[:, 5]
  v_tgt = obs_raw[:, 7]
  steer_q = obs_raw[:, 9]
  steer_qd = obs_raw[:, 10]
  e_lat = obs_raw[:, 13]
  e_psi = obs_raw[:, 14]
  k0 = obs_raw[:, 15]
  k1 = obs_raw[:, 16]
  k2 = obs_raw[:, 17]
  target_pitch = obs_raw[:, 19]
  wheelie = obs_raw[:, 20] > 0.5

  v_eff = np.maximum(1.6, np.abs(v_fwd))
  k_ff = 0.25 * k0 + 0.40 * k1 + 0.35 * k2

  # Speed-invariant overdamped vector-field intercept heading (handles both
  # tight road tracking and smooth U-turns from far off-track / opposite heading)
  l_look = np.where(
      np.abs(e_lat) < 1.2,
      np.maximum(2.8, 1.05 * v_eff),
      np.maximum(2.2, 0.78 * v_eff),
  )
  psi_des = np.arctan2(-e_lat, l_look)
  e_psi_eff = (e_psi - psi_des + np.pi) % (2.0 * np.pi) - np.pi
  on_track = np.exp(-0.25 * e_lat * e_lat)
  a_lat = np.clip(-1.7 * v_eff * e_psi_eff, -4.3, 4.3)
  k_des = np.clip(on_track * k_ff + a_lat / (v_eff * v_eff), -0.68, 0.68)
  phi_des = np.clip(-np.arctan((v_eff * v_eff * k_des) / 9.81), -0.45, 0.45)
  phi_des = np.where(wheelie, 0.30 * phi_des, phi_des)

  k_actual = -(9.81 * np.tan(np.clip(lean, -0.58, 0.58))) / (v_eff * v_eff)
  delta_actual = np.arctan(WHEELBASE * k_actual) / np.cos(HEAD_ANGLE_RAD)

  k_interp = np.column_stack(
      [np.interp(v_eff, lqr_speeds, lqr_k[:, i]) for i in range(4)]
  )
  x_err = np.column_stack(
      [lean - phi_des, lean_rate, steer_q - delta_actual, steer_qd]
  )
  steer_lqr = np.clip(
      delta_actual - np.einsum('ij,ij->i', k_interp, x_err), -0.70, 0.70
  )
  front_air = wheelie & (pitch < -0.04)
  steer_air = np.clip(-0.55 * lean - 0.10 * lean_rate, -0.25, 0.25)
  steer_cmd = np.where(front_air, steer_air, steer_lqr)

  omega_c_des = (v_tgt / WHEEL_RADIUS) / CHAIN_RATIO
  drive_cruise = np.clip(
      22.0 * (v_tgt - v_fwd) + 4.0 * (omega_c_des - crank_qvel), -140.0, 140.0
  )
  lean_safe = np.clip(1.0 - np.maximum(0.0, np.abs(lean) - 0.18) / 0.18, 0.0, 1.0)
  speed_safe = np.clip((5.9 - v_fwd) / 0.7, 0.0, 1.0)
  eff_target_pitch = target_pitch * lean_safe * speed_safe
  # Realistic wheelie feedforward (~145 Nm once pitched up, ~275 Nm pop burst)
  drive_wheelie = np.clip(
      145.0 * lean_safe * speed_safe + 480.0 * (pitch - eff_target_pitch) + 65.0 * pitch_rate,
      -40.0,
      300.0,
  )
  drive_cmd = np.where(wheelie, drive_wheelie, drive_cruise)

  pr_cmd = np.where(
      wheelie,
      np.clip(1.8 * lean + 0.32 * lean_rate, -0.35, 0.35),
      0.0,
  )
  pp_cmd = np.where(
      wheelie, np.clip(eff_target_pitch - 0.12, -0.36, 0.34), 0.04
  )

  phys = np.column_stack([steer_cmd, drive_cmd, pr_cmd, pp_cmd])
  norm = (phys - ACT_BIAS[None, :]) / ACT_SCALE[None, :]
  return np.clip(norm, -0.995, 0.995)


class PelicanPolicy:
  """Lightweight pure-NumPy 2-hidden-layer tanh MLP policy."""

  def __init__(
      self,
      obs_mean: np.ndarray,
      obs_std: np.ndarray,
      w1: np.ndarray,
      b1: np.ndarray,
      w2: np.ndarray,
      b2: np.ndarray,
      w3: np.ndarray,
      b3: np.ndarray,
      metadata: dict | None = None,
      lqr_speeds: np.ndarray | None = None,
      lqr_k: np.ndarray | None = None,
  ) -> None:
    self.obs_mean = np.asarray(obs_mean, dtype=np.float64)
    self.obs_std = np.maximum(1e-3, np.asarray(obs_std, dtype=np.float64))
    self.w1 = np.asarray(w1, dtype=np.float64)
    self.b1 = np.asarray(b1, dtype=np.float64)
    self.w2 = np.asarray(w2, dtype=np.float64)
    self.b2 = np.asarray(b2, dtype=np.float64)
    self.w3 = np.asarray(w3, dtype=np.float64)
    self.b3 = np.asarray(b3, dtype=np.float64)
    self.metadata = metadata or {}
    self.lqr_speeds = (
        np.asarray(lqr_speeds, dtype=np.float64) if lqr_speeds is not None else None
    )
    self.lqr_k = np.asarray(lqr_k, dtype=np.float64) if lqr_k is not None else None

  @classmethod
  def load(cls, path: str = DEFAULT_POLICY_PATH) -> PelicanPolicy:
    data = np.load(path, allow_pickle=False)
    meta = {}
    for k in ('iter', 'eval_err_m', 'eval_surv_s', 'eval_wheelie_deg'):
      if k in data:
        meta[k] = float(data[k])
    lqr_s = data['lqr_speeds'] if 'lqr_speeds' in data else None
    lqr_k = data['lqr_k'] if 'lqr_k' in data else None
    return cls(
        obs_mean=data['obs_mean'],
        obs_std=data['obs_std'],
        w1=data['w1'],
        b1=data['b1'],
        w2=data['w2'],
        b2=data['b2'],
        w3=data['w3'],
        b3=data['b3'],
        metadata=meta,
        lqr_speeds=lqr_s,
        lqr_k=lqr_k,
    )

  def act(self, obs: np.ndarray) -> np.ndarray:
    """Deterministic policy forward pass returning physical actuator commands."""
    x = np.clip((obs - self.obs_mean) / self.obs_std, -8.0, 8.0)
    h1 = np.tanh(x @ self.w1 + self.b1)
    h2 = np.tanh(h1 @ self.w2 + self.b2)
    raw = np.tanh(h2 @ self.w3 + self.b3)
    return raw * ACT_SCALE + ACT_BIAS
