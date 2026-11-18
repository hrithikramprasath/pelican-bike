"""Shared message definitions between sim_plugin.py and viewer_plugin.py."""

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class RideConfig(messages.Snapshot):
  """UI -> sim: live controller and scene parameters."""

  mode: str = 'figure8'  # 'cruise', 'figure8', 'circle_left', 'circle_right', 'manual'
  target_speed: float = 3.4  # m/s
  manual_steer: float = 0.0  # Desired curvature / turn command in [-1, 1]
  wing_mode: str = 'handlebars'  # 'handlebars', 'glide', 'flap'
  flap_intensity: float = 0.75
  beak_open_hold: bool = False
  auto_head_look: bool = True
  environment: str = 'coastal_road'  # 'coastal_road', 'slalom_cones', 'checker'
  follow_camera: bool = True
  use_learned_policy: bool = True


@dataclasses.dataclass(frozen=True)
class ActionEvent(messages.Event):
  """UI -> sim: discrete rider/bike action & trick triggers."""

  action: str = 'reset_ride'


@dataclasses.dataclass(frozen=True)
class RideTelemetry(messages.Snapshot):
  """sim -> UI: live bicycle and pelican telemetry."""

  sim_time: float = 0.0
  speed_mps: float = 0.0
  target_speed_mps: float = 3.4
  lean_deg: float = 0.0
  pitch_deg: float = 0.0
  steer_deg: float = 0.0
  crank_rpm: float = 0.0
  distance_m: float = 0.0
  road_err_m: float = 0.0
  pos_x: float = 0.0
  pos_y: float = 0.0
  heading_deg: float = 0.0
  upright: bool = True
  airborne: bool = False
  beak_deg: float = 0.0
  mode: str = 'figure8'
  active_trick: str = ''
  pending_trick: str = ''
  trick_count: int = 0
  policy_label: str = 'Trained PPO MLP'
