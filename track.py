"""Road geometry and continuous phase-locked path tracking for Pelican Bike.

Provides a unified local road-frame representation
``(e_lat, e_psi, kappa_0, kappa_1, kappa_2, kappa_3)`` across Figure-8,
left/right circles, straight cruise, and manual steer modes. Uses local
arc-length phase tracking with heading disambiguation so the Figure-8 crossover
at ``(0, 0)`` never causes branch jumps.
"""

from __future__ import annotations

import math
import numpy as np

FIG8_A = 7.5
FIG8_B = 3.8
FIG8_START_YAW = math.atan2(2.0 * FIG8_B, FIG8_A)
CIRCLE_RADIUS = 4.6

_N_FIG8 = 720
_S_GRID = np.linspace(0.0, 2.0 * math.pi, _N_FIG8, endpoint=False)
_DX = FIG8_A * np.cos(_S_GRID)
_DY = 2.0 * FIG8_B * np.cos(2.0 * _S_GRID)
_DDX = -FIG8_A * np.sin(_S_GRID)
_DDY = -4.0 * FIG8_B * np.sin(2.0 * _S_GRID)
_SPEED_DS = np.hypot(_DX, _DY)
_DS = (2.0 * math.pi) / _N_FIG8

FIG8_PTS = np.column_stack(
    [FIG8_A * np.sin(_S_GRID), FIG8_B * np.sin(2.0 * _S_GRID)]
)
FIG8_YAWS = np.arctan2(_DY, _DX)
FIG8_TANGENTS = np.column_stack([np.cos(FIG8_YAWS), np.sin(FIG8_YAWS)])
FIG8_KAPPAS = (_DX * _DDY - _DY * _DDX) / np.maximum(1e-6, _SPEED_DS**3)
FIG8_ARC_LEN = float(np.sum(_SPEED_DS) * _DS)

# Cumulative arc length at each grid index (m)
_ARC_SEG = _SPEED_DS * _DS
FIG8_CUM_ARC = np.concatenate([[0.0], np.cumsum(_ARC_SEG[:-1])])


def _build_lookahead_index_table(dist_m: float) -> np.ndarray:
  """Precomputes grid index offset corresponding to `dist_m` forward."""
  offsets = np.zeros(_N_FIG8, dtype=np.int32)
  for i in range(_N_FIG8):
    acc = 0.0
    step = 0
    while acc < dist_m and step < _N_FIG8 // 2:
      acc += float(_ARC_SEG[(i + step) % _N_FIG8])
      step += 1
    offsets[i] = (i + step) % _N_FIG8
  return offsets


FIG8_LOOK_1 = _build_lookahead_index_table(1.1)
FIG8_LOOK_2 = _build_lookahead_index_table(2.3)
FIG8_LOOK_3 = _build_lookahead_index_table(3.7)
FIG8_LOOK_4 = _build_lookahead_index_table(5.5)


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def match_fig8_index(
    px: float,
    py: float,
    yaw: float,
    prev_idx: int | None = None,
    window: int = 38,
) -> int:
  """Finds the phase-consistent nearest index on the Figure-8 track."""
  if prev_idx is None:
    diffs = FIG8_PTS - np.array([px, py])
    d2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # When far off track (> 2.0 m), pick the spatially closest point so off-track
    # recovery drives straight back to the nearest road segment. Near the road
    # (<= 2.0 m), disambiguate the (0, 0) crossover using heading alignment.
    if float(np.min(d2)) > 4.0:
      return int(np.argmin(d2))
    yaw_cost = 0.8 * (1.0 - np.cos(FIG8_YAWS - yaw))
    return int(np.argmin(d2 + yaw_cost))

  # Search only in a local arc window around prev_idx to eliminate crossover jumps
  offsets = np.arange(-window, window + 1)
  cand_idx = (prev_idx + offsets) % _N_FIG8
  pts = FIG8_PTS[cand_idx]
  dx = pts[:, 0] - px
  dy = pts[:, 1] - py
  d2 = dx * dx + dy * dy
  yaw_cost = 0.45 * (1.0 - np.cos(FIG8_YAWS[cand_idx] - yaw))
  best_local = int(np.argmin(d2 + yaw_cost))
  # If bike wandered far off the local arc (> 3.0 m), re-acquire globally
  if d2[best_local] > 9.0:
    return match_fig8_index(px, py, yaw, prev_idx=None)
  return int(cand_idx[best_local])


def fig8_features(
    px: float,
    py: float,
    yaw: float,
    prev_idx: int | None = None,
) -> tuple[int, float, float, float, float, float, float, float]:
  """Returns (s_idx, e_lat, e_psi, k0, k1, k2, k3, k4) on Figure-8."""
  idx = match_fig8_index(px, py, yaw, prev_idx=prev_idx)
  rx, ry = float(FIG8_PTS[idx, 0]), float(FIG8_PTS[idx, 1])
  tx, ty = float(FIG8_TANGENTS[idx, 0]), float(FIG8_TANGENTS[idx, 1])
  dx = px - rx
  dy = py - ry
  # Signed lateral error (+ is left of path tangent)
  e_lat = -dx * ty + dy * tx
  e_psi = float(wrap_pi(yaw - float(FIG8_YAWS[idx])))
  k0 = float(FIG8_KAPPAS[idx])
  k1 = float(FIG8_KAPPAS[FIG8_LOOK_1[idx]])
  k2 = float(FIG8_KAPPAS[FIG8_LOOK_2[idx]])
  k3 = float(FIG8_KAPPAS[FIG8_LOOK_3[idx]])
  k4 = float(FIG8_KAPPAS[FIG8_LOOK_4[idx]])
  return idx, e_lat, e_psi, k0, k1, k2, k3, k4


def fig8_features_batch(
    px: np.ndarray,
    py: np.ndarray,
    yaw: np.ndarray,
    prev_idx: np.ndarray,
    window: int = 38,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Vectorized Figure-8 local phase tracking for Batch environments."""
  offsets = np.arange(-window, window + 1)  # (W,)
  cand_idx = (prev_idx[:, None] + offsets[None, :]) % _N_FIG8  # (N, W)
  pts = FIG8_PTS[cand_idx]  # (N, W, 2)
  dx = pts[:, :, 0] - px[:, None]
  dy = pts[:, :, 1] - py[:, None]
  d2 = dx * dx + dy * dy
  yaw_cost = 0.45 * (1.0 - np.cos(FIG8_YAWS[cand_idx] - yaw[:, None]))
  best_local = np.argmin(d2 + yaw_cost, axis=1)
  idx = cand_idx[np.arange(len(prev_idx)), best_local]
  far = d2[np.arange(len(prev_idx)), best_local] > 9.0
  if np.any(far):
    for eid in np.where(far)[0]:
      idx[eid] = match_fig8_index(float(px[eid]), float(py[eid]), float(yaw[eid]), prev_idx=None)

  rx = FIG8_PTS[idx, 0]
  ry = FIG8_PTS[idx, 1]
  tx = FIG8_TANGENTS[idx, 0]
  ty = FIG8_TANGENTS[idx, 1]
  ex = px - rx
  ey = py - ry
  e_lat = -ex * ty + ey * tx
  e_psi = wrap_pi(yaw - FIG8_YAWS[idx])
  k0 = FIG8_KAPPAS[idx]
  k1 = FIG8_KAPPAS[FIG8_LOOK_1[idx]]
  k2 = FIG8_KAPPAS[FIG8_LOOK_2[idx]]
  k3 = FIG8_KAPPAS[FIG8_LOOK_3[idx]]
  k4 = FIG8_KAPPAS[FIG8_LOOK_4[idx]]
  return idx, e_lat, e_psi, k0, k1, k2, k3, k4


def circle_features(
    px: float,
    py: float,
    yaw: float,
    direction: float = 1.0,
    radius: float = CIRCLE_RADIUS,
) -> tuple[float, float, float, float, float, float]:
  """Returns (e_lat, e_psi, k0, k1, k2, k3) for CCW (+1) or CW (-1) circle."""
  r = max(1e-4, math.hypot(px, py))
  radial_ang = math.atan2(py, px)
  tangent_yaw = radial_ang + direction * 0.5 * math.pi
  # Signed lateral error (+ is to the LEFT of travel direction)
  # For CCW (+1), LEFT points inward (-radial), so inside circle => e_lat > 0.
  # For CW (-1), LEFT points outward (+radial), so outside circle => e_lat > 0.
  if direction >= 0.0:
    e_lat = radius - r
  else:
    e_lat = r - radius
  e_psi = float(wrap_pi(yaw - tangent_yaw))
  k = direction / radius
  return e_lat, e_psi, k, k, k, k


def cruise_features(
    px: float,
    py: float,
    yaw: float,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
) -> tuple[float, float, float, float, float, float]:
  """Returns (e_lat, e_psi, 0, 0, 0, 0) for straight-line tracking."""
  tx, ty = math.cos(anchor_yaw), math.sin(anchor_yaw)
  dx = px - anchor_x
  dy = py - anchor_y
  e_lat = -dx * ty + dy * tx
  e_psi = float(wrap_pi(yaw - anchor_yaw))
  return e_lat, e_psi, 0.0, 0.0, 0.0, 0.0


def safe_corner_speed(
    target_speed: float,
    k0: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float = 0.0,
    max_lat_accel: float = 4.4,
) -> float:
  """Reduces target speed smoothly into tight corners to prevent tire slipout & pedal strike."""
  max_k = max(abs(k0), abs(k1), abs(k2), abs(k3), abs(k4))
  if max_k < 1e-4:
    return target_speed
  # Keep lateral centripetal accel <= ~4.4 m/s^2 (<= ~24 deg lean, well clear of 31 deg pedal strike)
  v_curve_cap = math.sqrt(max_lat_accel / max_k)
  return max(1.8, min(target_speed, v_curve_cap))
