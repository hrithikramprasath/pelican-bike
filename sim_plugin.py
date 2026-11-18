"""Simulation-side Studio plugin for the Pelican Riding a Bicycle simulation.

Builds the coastal boardwalk / slalom / studio scene procedurally via ``MjSpec``,
attaches the articulated ``assets/pelican_bicycle/model.xml`` asset, and runs
the trained PPO neural-network rider policy (``trained_policy.npz``) with
continuous phase-locked road tracking and physically simulated button tricks
(Turbo Boost, Wheelie, Wing-Flap Hop, Stoppie, No-Hands, and S-Carve).
"""

from __future__ import annotations

import math
import os

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

import policy as policy_mod
import track

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(WORKSPACE_DIR, 'assets', 'pelican_bicycle')
MODEL_XML = os.path.join(ASSET_DIR, 'model.xml')
SCENE_XML = os.path.join(WORKSPACE_DIR, 'scene.xml')

BIKE_PREFIX = 'pb_'
WHEEL_RADIUS = 0.330
WHEELBASE = 1.090
HEAD_ANGLE_RAD = math.radians(20.0)
CHAIN_RATIO = 2.0
POLICY_SUBSTEPS = 10  # 50 Hz control (10 * 0.002s = 0.02s)

TRICK_LABELS = {
    'turbo_boost': 'TURBO BOOST',
    'wheelie': 'WHEELIE',
    'wing_hop': 'WING-FLAP HOP',
    'stoppie': 'STOPPIE',
    'no_hands': 'NO-HANDS',
    's_carve': 'S-CARVE',
}


def _quat_from_yaw(yaw: float) -> list[float]:
  return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


class SimPlugin(plugin.SimPlugin):
  """Builds the pelican-on-bicycle scene and runs the trained rider policy."""

  # Live controller / scene config (class-level defaults survive hot-reload)
  mode: str = 'figure8'
  target_speed: float = 3.4
  manual_steer: float = 0.0
  wing_mode: str = 'handlebars'
  flap_intensity: float = 0.75
  beak_open_hold: bool = False
  auto_head_look: bool = True
  environment: str = 'coastal_road'
  use_learned_policy: bool = True

  # Internal sim state
  spec: mujoco.MjSpec | None = None
  policy: policy_mod.PelicanPolicy | None = None
  lqr_speeds: np.ndarray | None = None
  lqr_k: np.ndarray | None = None
  reset_requested: bool = True
  initialized_once: bool = False
  track_s_idx: int | None = None
  cruise_anchor_x: float = 0.0
  cruise_anchor_y: float = 0.0
  cruise_anchor_yaw: float = 0.0

  # Action / trick timers
  active_trick: str = ''
  pending_trick: str = ''
  trick_start: float = -1.0
  trick_pop_at: float = -1.0
  trick_until: float = -1.0
  stoppie_released: bool = False
  wheelie_landed: bool = False
  trick_count: int = 0
  flap_burst_until: float = -1.0
  beak_snap_until: float = -1.0
  bell_nod_until: float = -1.0

  # Telemetry / step counters
  distance_m: float = 0.0
  prev_xy_valid: bool = False
  prev_x: float = 0.0
  prev_y: float = 0.0
  road_err_m: float = 0.0
  step_counter: int = 0
  telemetry_counter: int = 0

  # Held 50 Hz policy commands
  last_steer_cmd: float = 0.0
  last_drive_cmd: float = 0.0
  last_brake_cmd: float = 0.0
  last_pr_cmd: float = 0.0
  last_pp_cmd: float = 0.0
  policy_mtime: float = 0.0

  def _ensure_policy(self, *, force: bool = False) -> None:
    if not os.path.exists(policy_mod.DEFAULT_POLICY_PATH):
      return
    try:
      mtime = os.path.getmtime(policy_mod.DEFAULT_POLICY_PATH)
    except OSError:
      return
    if self.policy is None or force or mtime != self.policy_mtime:
      try:
        self.policy = policy_mod.PelicanPolicy.load(policy_mod.DEFAULT_POLICY_PATH)
        self.policy_mtime = mtime
        if self.policy.lqr_speeds is not None and self.policy.lqr_k is not None:
          self.lqr_speeds = self.policy.lqr_speeds
          self.lqr_k = self.policy.lqr_k
      except Exception:
        pass

  def _ensure_lqr_fallback(self) -> None:
    if self.lqr_speeds is not None and self.lqr_k is not None:
      return
    self._ensure_policy()
    if self.lqr_speeds is None or self.lqr_k is None:
      import train_policy  # fallback only if npz lacks LQR table
      m_clean = train_policy.build_training_model(with_posts=False)
      self.lqr_speeds, self.lqr_k = train_policy.compute_inner_lqr_table(m_clean)

  def _add_coastal_scenery(self, spec: mujoco.MjSpec) -> None:
    """Adds non-colliding smooth figure-8 boardwalk road mesh and perimeter posts."""
    world = spec.worldbody

    n_seg = 180
    half_w = 0.96
    s_vals = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    verts = []
    faces = []
    for s in s_vals:
      sf = float(s)
      cx = track.FIG8_A * math.sin(sf)
      cy = track.FIG8_B * math.sin(2.0 * sf)
      tx = track.FIG8_A * math.cos(sf)
      ty = 2.0 * track.FIG8_B * math.cos(2.0 * sf)
      norm = max(1e-6, math.hypot(tx, ty))
      nx, ny = -ty / norm, tx / norm
      z_top = 0.0050 + 0.0005 * math.cos(sf)
      verts.append([cx - half_w * nx, cy - half_w * ny, z_top])
      verts.append([cx + half_w * nx, cy + half_w * ny, z_top])
      verts.append([cx - half_w * nx, cy - half_w * ny, 0.001])
      verts.append([cx + half_w * nx, cy + half_w * ny, 0.001])

    for i in range(n_seg):
      jn = (i + 1) % n_seg
      b0, b1 = 4 * i, 4 * jn
      faces.append([b0 + 0, b1 + 1, b0 + 1])
      faces.append([b0 + 0, b1 + 0, b1 + 1])
      faces.append([b0 + 0, b0 + 2, b1 + 2])
      faces.append([b0 + 0, b1 + 2, b1 + 0])
      faces.append([b0 + 1, b1 + 1, b1 + 3])
      faces.append([b0 + 1, b1 + 3, b0 + 3])
      faces.append([b0 + 2, b0 + 3, b1 + 3])
      faces.append([b0 + 2, b1 + 3, b1 + 2])

    v_arr = np.asarray(verts, dtype=np.float32).flatten()
    f_arr = np.asarray(faces, dtype=np.int32).flatten()
    spec.add_mesh(
        name='fig8_road_mesh',
        uservert=v_arr,
        userface=f_arr,
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    world.add_geom(
        name='fig8_road_geom',
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname='fig8_road_mesh',
        rgba=[0.22, 0.25, 0.29, 1.0],
        contype=0,
        conaffinity=0,
        group=1,
        mass=0.0,
    )

    n_dashes = 90
    s_dash = np.linspace(0.0, 2.0 * math.pi, n_dashes, endpoint=False)
    for idx, s in enumerate(s_dash):
      if idx % 2 != 0:
        continue
      sf = float(s)
      cx = track.FIG8_A * math.sin(sf)
      cy = track.FIG8_B * math.sin(2.0 * sf)
      tx = track.FIG8_A * math.cos(sf)
      ty = 2.0 * track.FIG8_B * math.cos(2.0 * sf)
      ang = math.atan2(ty, tx)
      z_dash = 0.0066 + 0.0005 * math.cos(sf)
      world.add_geom(
          name=f'road_dash_{idx}',
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=[cx, cy, z_dash],
          quat=_quat_from_yaw(ang),
          size=[0.22, 0.042, 0.0010],
          rgba=[0.95, 0.80, 0.25, 1.0],
          contype=0,
          conaffinity=0,
          group=1,
          mass=0.0,
      )

    n_posts = 20
    for i in range(n_posts):
      th = 2.0 * math.pi * i / n_posts
      px = 11.2 * math.cos(th)
      py = 7.2 * math.sin(th)
      post = world.add_body(name=f'coastal_post_{i}', pos=[px, py, 0.0])
      post.add_geom(
          name=f'post_wood_{i}',
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=[0.0, 0.0, 0.42],
          size=[0.075, 0.42],
          rgba=[0.52, 0.36, 0.22, 1.0],
          group=1,
      )
      post.add_geom(
          name=f'post_cap_{i}',
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          pos=[0.0, 0.0, 0.86],
          size=[0.088],
          rgba=[0.18, 0.66, 0.62, 1.0] if i % 2 == 0 else [0.95, 0.56, 0.18, 1.0],
          group=1,
      )

  def _add_slalom_cones(self, spec: mujoco.MjSpec) -> None:
    """Adds knockable physics traffic cones inside the arena."""
    world = spec.worldbody
    cone_positions = [
        (3.2, 1.6),
        (5.5, 0.0),
        (3.2, -1.6),
        (-3.2, 1.6),
        (-5.5, 0.0),
        (-3.2, -1.6),
        (0.0, 3.2),
        (0.0, -3.2),
    ]
    for i, (cx, cy) in enumerate(cone_positions):
      cone = world.add_body(name=f'traffic_cone_{i}', pos=[cx, cy, 0.0])
      cone.add_freejoint(name=f'cone_free_{i}')
      cone.add_geom(
          name=f'cone_base_{i}',
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=[0, 0, 0.015],
          size=[0.16, 0.16, 0.015],
          rgba=[0.92, 0.36, 0.08, 1.0],
          mass=0.8,
          condim=4,
          friction=[0.7, 0.02, 0.002],
          group=1,
      )
      cone.add_geom(
          name=f'cone_lower_{i}',
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=[0, 0, 0.13],
          size=[0.095, 0.10],
          rgba=[0.95, 0.42, 0.10, 1.0],
          mass=0.4,
          group=1,
      )
      cone.add_geom(
          name=f'cone_stripe_{i}',
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=[0, 0, 0.28],
          size=[0.068, 0.06],
          rgba=[0.96, 0.96, 0.96, 1.0],
          mass=0.2,
          group=1,
      )
      cone.add_geom(
          name=f'cone_tip_{i}',
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=[0, 0, 0.40],
          size=[0.040, 0.065],
          rgba=[0.95, 0.42, 0.10, 1.0],
          mass=0.1,
          group=1,
      )

  def build_spec(self, *, portable_paths: bool = False) -> mujoco.MjSpec:
    """Constructs the simulation scene and attaches the pelican bicycle."""
    spec = mujoco.MjSpec()
    spec.modelname = 'pelican_riding_bicycle'
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0

    if portable_paths:
      spec.meshdir = 'assets/pelican_bicycle/meshes'
      spec.texturedir = 'assets/pelican_bicycle/textures'
    else:
      spec.meshdir = os.path.join(ASSET_DIR, 'meshes')
      spec.texturedir = os.path.join(ASSET_DIR, 'textures')

    spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.25, 0.25, 0.25]
    spec.visual.rgba.haze = [0.58, 0.78, 0.94, 1.0]
    spec.visual.global_.azimuth = 135.0
    spec.visual.global_.elevation = -18.0
    spec.stat.extent = 4.2

    spec.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.52, 0.78, 0.96],
        rgb2=[0.18, 0.38, 0.62],
        width=512,
        height=2048,
    )
    if self.environment == 'coastal_road':
      rgb1 = [0.82, 0.75, 0.60]
      rgb2 = [0.76, 0.69, 0.54]
      markrgb = [0.88, 0.82, 0.68]
    elif self.environment == 'slalom_cones':
      rgb1 = [0.26, 0.30, 0.34]
      rgb2 = [0.22, 0.26, 0.30]
      markrgb = [0.55, 0.58, 0.62]
    else:
      rgb1 = [0.22, 0.32, 0.42]
      rgb2 = [0.14, 0.22, 0.32]
      markrgb = [0.75, 0.80, 0.85]

    spec.add_texture(
        name='ground_tex',
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=rgb1,
        rgb2=rgb2,
        markrgb=markrgb,
        width=300,
        height=300,
    )
    ground_mat = spec.add_material(
        name='ground_mat', texrepeat=[2.0, 2.0], texuniform=True, reflectance=0.02
    )
    ground_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_tex'

    world = spec.worldbody
    sun = world.add_light(
        pos=[4.0, -5.0, 12.0],
        dir=[-0.25, 0.35, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True,
    )
    sun.diffuse = [0.95, 0.92, 0.85]
    fill = world.add_light(
        pos=[-6.0, 6.0, 8.0],
        dir=[0.4, -0.4, -0.8],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=False,
    )
    fill.diffuse = [0.35, 0.45, 0.60]

    world.add_geom(
        name='ground_plane',
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material='ground_mat',
        condim=6,
        friction=[1.2, 0.01, 0.001],
        group=1,
    )

    if self.environment == 'coastal_road':
      self._add_coastal_scenery(spec)
    elif self.environment == 'slalom_cones':
      self._add_slalom_cones(spec)

    bike_spec = mujoco.MjSpec.from_file(MODEL_XML)
    bike_root = bike_spec.worldbody.first_body()
    bike_root.add_freejoint(name='free')

    start_yaw = track.FIG8_START_YAW if self.mode == 'figure8' else 0.0
    spawn_frame = world.add_frame(pos=[0.0, 0.0, 0.0], quat=_quat_from_yaw(start_yaw))
    spec.attach(bike_spec, frame=spawn_frame, prefix=BIKE_PREFIX)
    return spec

  def _initialize_rolling_state(
      self, model: mujoco.MjModel, data: mujoco.MjData
  ) -> None:
    """Places the bicycle on its start pose with matched rolling wheel velocity."""
    mujoco.mj_resetData(model, data)
    start_yaw = track.FIG8_START_YAW if self.mode == 'figure8' else 0.0
    v0 = max(1.8, float(self.target_speed))
    vx = v0 * math.cos(start_yaw)
    vy = v0 * math.sin(start_yaw)

    free_jnt = data.joint(f'{BIKE_PREFIX}free')
    free_jnt.qpos[0:3] = [0.0, 0.0, 0.0]
    free_jnt.qpos[3:7] = _quat_from_yaw(start_yaw)
    free_jnt.qvel[0:3] = [vx, vy, 0.0]

    omega_w = v0 / WHEEL_RADIUS
    data.joint(f'{BIKE_PREFIX}rear_wheel_joint').qvel[0] = omega_w
    data.joint(f'{BIKE_PREFIX}front_wheel_joint').qvel[0] = omega_w
    data.joint(f'{BIKE_PREFIX}crank_joint').qvel[0] = omega_w / CHAIN_RATIO
    data.xfrc_applied[:] = 0.0

    mujoco.mj_forward(model, data)
    self.track_s_idx = 0 if self.mode == 'figure8' else None
    self.cruise_anchor_x = 0.0
    self.cruise_anchor_y = 0.0
    self.cruise_anchor_yaw = start_yaw
    self.active_trick = ''
    self.pending_trick = ''
    self.trick_start = -1.0
    self.trick_pop_at = -1.0
    self.trick_until = -1.0
    self.stoppie_released = False
    self.wheelie_landed = False
    self.flap_burst_until = -1.0
    self.beak_snap_until = -1.0
    self.bell_nod_until = -1.0
    self.distance_m = 0.0
    self.road_err_m = 0.0
    self.step_counter = 0
    self.last_steer_cmd = 0.0
    self.last_drive_cmd = 0.0
    self.last_brake_cmd = 0.0
    self.last_pr_cmd = 0.0
    self.last_pp_cmd = 0.04
    self.prev_xy_valid = True
    pos = data.body(f'{BIKE_PREFIX}pelican_bicycle').xpos
    self.prev_x = float(pos[0])
    self.prev_y = float(pos[1])

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    if event.path:
      return False
    self._ensure_policy()
    self.spec = self.build_spec()
    self.handle.set_model(self.spec, watch=[MODEL_XML])
    return True

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    self._ensure_policy()
    if self.reset_requested or not self.initialized_once:
      self.reset_requested = False
      self.initialized_once = True
      self._initialize_rolling_state(event.model, event.data)
    self.telemetry_counter = 0

  @messages.handler
  def on_reset(self, event: messages.ResetEvent) -> None:
    del event
    if self.handle.model is not None and self.handle.data is not None:
      self._initialize_rolling_state(self.handle.model, self.handle.data)

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins:
      self._ensure_policy(force=True)
      d = self.handle.data
      if d is not None:
        try:
          up_z = float(d.body(f'{BIKE_PREFIX}pelican_bicycle').xmat.reshape(3, 3)[2, 2])
          if up_z < 0.45:
            self.reset_requested = True
        except Exception:
          pass
      self.handle.reload()

  @messages.handler
  def on_ride_config(self, snapshot: user_messages.RideConfig) -> None:
    env_changed = snapshot.environment != self.environment
    mode_changed = snapshot.mode != self.mode
    self.mode = snapshot.mode
    self.target_speed = float(snapshot.target_speed)
    self.manual_steer = float(snapshot.manual_steer)
    self.wing_mode = snapshot.wing_mode
    self.flap_intensity = float(snapshot.flap_intensity)
    self.beak_open_hold = bool(snapshot.beak_open_hold)
    self.auto_head_look = bool(snapshot.auto_head_look)
    self.environment = snapshot.environment
    self.use_learned_policy = bool(getattr(snapshot, 'use_learned_policy', True))
    if env_changed:
      self.reset_requested = True
      self.handle.reload()
    elif mode_changed:
      self.track_s_idx = None
      if self.mode == 'cruise' and self.handle.data is not None:
        pos = self.handle.data.body(f'{BIKE_PREFIX}pelican_bicycle').xpos
        xmat = self.handle.data.body(f'{BIKE_PREFIX}pelican_bicycle').xmat.reshape(3, 3)
        fwd = xmat[:, 0]
        self.cruise_anchor_x = float(pos[0])
        self.cruise_anchor_y = float(pos[1])
        self.cruise_anchor_yaw = math.atan2(float(fwd[1]), float(fwd[0]))

  def _trigger_trick(self, name: str, t_now: float) -> None:
    durations = {
        'turbo_boost': 5.0,
        'wheelie': 8.5,
        'wing_hop': 7.5,
        'stoppie': 8.5,
        'no_hands': 3.8,
        's_carve': 4.2,
    }
    snap_durs = {
        'turbo_boost': 1.0,
        'wheelie': 0.8,
        'wing_hop': 0.8,
        'stoppie': 0.7,
        'no_hands': 1.2,
        's_carve': 0.0,
    }
    self.active_trick = name
    self.pending_trick = ''
    self.trick_start = t_now
    self.trick_pop_at = -1.0 if name in ('wheelie', 'stoppie', 'wing_hop') else t_now
    self.trick_until = t_now + durations[name]
    self.stoppie_released = False
    self.wheelie_landed = False
    sd = snap_durs.get(name, 0.0)
    if sd > 0.0:
      self.beak_snap_until = max(self.beak_snap_until, t_now + sd)

  @messages.handler
  def on_action(self, event: user_messages.ActionEvent) -> None:
    data = self.handle.data
    model = self.handle.model
    if data is None or model is None:
      return
    t_now = float(data.time)
    act = event.action
    p = BIKE_PREFIX
    if act in TRICK_LABELS:
      front_lift = float(data.body(f'{p}front_wheel').xpos[2]) - WHEEL_RADIUS
      rear_lift = float(data.body(f'{p}rear_wheel').xpos[2]) - WHEEL_RADIUS
      xmat = data.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
      pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))
      is_settled = front_lift < 0.030 and rear_lift < 0.030 and abs(pitch) < 0.08
      mid_pop_busy = (
          self.active_trick in ('wheelie', 'stoppie', 'wing_hop')
          and self.trick_pop_at >= 0.0
          and not self.wheelie_landed
          and not self.stoppie_released
      )
      if not is_settled or mid_pop_busy:
        self.pending_trick = act
      else:
        self._trigger_trick(act, t_now)
    elif act == 'flap_burst':
      self.flap_burst_until = t_now + 2.6
      self.beak_snap_until = max(self.beak_snap_until, t_now + 0.9)
    elif act == 'snap_beak':
      self.beak_snap_until = t_now + 0.85
    elif act == 'ring_bell':
      self.bell_nod_until = t_now + 0.65
      self.beak_snap_until = max(self.beak_snap_until, t_now + 0.50)
    elif act in ('nudge_left', 'nudge_right'):
      xmat = data.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
      left_vec = xmat[:, 1]
      sign = 1.0 if act == 'nudge_left' else -1.0
      free_qvel = data.joint(f'{p}free').qvel
      free_qvel[0] += sign * 0.55 * float(left_vec[0])
      free_qvel[1] += sign * 0.55 * float(left_vec[1])
      free_qvel[3] -= sign * 0.65
    elif act == 'reset_ride':
      self._initialize_rolling_state(model, data)
    elif act == 'bake_xml':
      portable = self.build_spec(portable_paths=True)
      with open(SCENE_XML, 'w', encoding='utf-8') as f:
        f.write(portable.to_xml())
      self.handle.dispatch(messages.LoadModelEvent(path=SCENE_XML))

  def _compute_road_features(
      self, px: float, py: float, yaw: float, t_now: float = 0.0
  ) -> tuple[float, float, float, float, float, float, float]:
    """Returns (e_lat, e_psi, k0, k1, k2, k3, k4) for active ride mode."""
    if self.mode == 'figure8':
      idx, e_lat, e_psi, k0, k1, k2, k3, k4 = track.fig8_features(
          px, py, yaw, prev_idx=self.track_s_idx
      )
      self.track_s_idx = idx
      return e_lat, e_psi, k0, k1, k2, k3, k4
    if self.mode == 'circle_left':
      el, ep, k0, k1, k2, k3 = track.circle_features(px, py, yaw, direction=1.0)
      return el, ep, k0, k1, k2, k3, k3
    if self.mode == 'circle_right':
      el, ep, k0, k1, k2, k3 = track.circle_features(px, py, yaw, direction=-1.0)
      return el, ep, k0, k1, k2, k3, k3
    if self.mode == 'manual':
      k_man = 0.48 * max(-1.0, min(1.0, self.manual_steer))
      if self.active_trick == 's_carve':
        elapsed = max(0.0, t_now - self.trick_start)
        env = 0.5 * (1.0 - math.cos(2.0 * math.pi * min(1.0, elapsed / 4.2)))
        k_man = max(
            -0.55,
            min(0.55, k_man + 0.34 * env * math.sin(2.0 * math.pi * 0.55 * elapsed)),
        )
      elif (
          self.active_trick in ('wheelie', 'stoppie', 'wing_hop')
          and self.trick_pop_at < 0.0
      ):
        k_man *= 0.15
      return 0.0, 0.0, k_man, k_man, k_man, k_man, k_man
    el, ep, k0, k1, k2, k3 = track.cruise_features(
        px, py, yaw, self.cruise_anchor_x, self.cruise_anchor_y, self.cruise_anchor_yaw
    )
    return el, ep, k0, k1, k2, k3, k3

  def _compute_lqr_action(
      self, obs: np.ndarray, crank_qvel: float
  ) -> np.ndarray:
    """Computes LQR fallback action in physical actuator units."""
    self._ensure_lqr_fallback()
    norm = policy_mod.lqr_supervisor_batch(
        obs[None, :],
        np.array([crank_qvel], dtype=np.float64),
        self.lqr_speeds,
        self.lqr_k,
    )[0]
    return norm * policy_mod.ACT_SCALE + policy_mod.ACT_BIAS

  @messages.handler
  def on_pre_step(self, event: messages.PreStepEvent) -> None:
    model = event.model
    data = event.data
    t = float(data.time)
    p = BIKE_PREFIX

    # Clear any transient aerodynamic forces from previous step
    data.xfrc_applied[:] = 0.0

    pos = data.body(f'{p}pelican_bicycle').xpos
    xmat = data.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
    fwd = xmat[:, 0]
    up = xmat[:, 2]
    yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    left_h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    lean = math.atan2(-float(np.dot(up, left_h)), max(1e-4, float(up[2])))
    pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))

    vel = data.sensor(f'{p}bike_vel').data
    gyro = data.sensor(f'{p}bike_gyro').data
    v_fwd = float(vel[0])
    v_lat = float(vel[1])
    lean_rate = float(gyro[0])
    pitch_rate = float(gyro[1])
    yaw_rate = float(gyro[2])

    upright = abs(lean) < 1.00 and float(up[2]) > 0.45
    front_lift = float(data.body(f'{p}front_wheel').xpos[2]) - WHEEL_RADIUS
    rear_lift = float(data.body(f'{p}rear_wheel').xpos[2]) - WHEEL_RADIUS
    grounded_settled = front_lift < 0.028 and rear_lift < 0.028 and abs(pitch) < 0.085

    # If fallen beyond recovery, relax drive & brakes so it settles naturally
    if not upright:
      self.active_trick = ''
      self.pending_trick = ''
      data.actuator(f'{p}drive_motor').ctrl[0] = 0.0
      data.actuator(f'{p}front_brake').ctrl[0] = 0.0
      return

    # Check trick completion (wait for touchdown before clearing airborne tricks!)
    if self.active_trick:
      early_done = (
          self.pending_trick != ''
          and self.trick_pop_at >= 0.0
          and (
              (self.active_trick == 'wheelie' and self.wheelie_landed)
              or (self.active_trick == 'stoppie' and self.stoppie_released)
              or (self.active_trick == 'wing_hop' and (t - self.trick_pop_at) > 0.65)
          )
          and grounded_settled
      )
      time_done = t >= self.trick_until and (
          self.active_trick not in ('wheelie', 'stoppie', 'wing_hop')
          or grounded_settled
          or t >= self.trick_until + 0.6
      )
      if early_done or time_done:
        self.trick_count += 1
        self.active_trick = ''

    if not self.active_trick and self.pending_trick and grounded_settled:
      self._trigger_trick(self.pending_trick, t)

    e_lat_raw, e_psi, k0, k1, k2, k3, k4 = self._compute_road_features(
        float(pos[0]), float(pos[1]), yaw, t_now=t
    )
    self.road_err_m = abs(e_lat_raw)

    trick = self.active_trick
    trick_elapsed = max(0.0, t - self.trick_start) if trick else 0.0

    # Trick-adjusted path/speed targets
    e_lat = e_lat_raw
    v_req = float(self.target_speed)
    if trick == 'turbo_boost':
      v_req = max(v_req, 5.8)
    elif trick == 's_carve':
      env = 0.5 * (1.0 - math.cos(2.0 * math.pi * min(1.0, trick_elapsed / 4.2)))
      e_lat = e_lat_raw - 0.55 * env * math.sin(2.0 * math.pi * 0.55 * trick_elapsed)

    if trick == 'turbo_boost':
      max_kb = max(abs(k0), abs(k1), abs(k2), 0.80 * abs(k3), 0.55 * abs(k4))
      v_tgt = max(2.1, min(v_req, math.sqrt(5.0 / max_kb))) if max_kb > 1e-4 else v_req
    else:
      v_tgt = track.safe_corner_speed(v_req, k0, k1, k2, k3, k4)
      if self.mode == 'manual' and abs(k0) > 0.22:
        v_tgt = max(v_tgt, 2.20)

    # Wait briefly for an upright/low-lean segment before popping Wheelie, Stoppie, or Wing-Hop
    if trick in ('wheelie', 'stoppie', 'wing_hop') and self.trick_pop_at < 0.0:
      if trick == 'wheelie':
        if abs(k1) < 0.32 and abs(k2) < 0.34:
          v_tgt = min(v_tgt, 2.15)
      elif trick == 'stoppie':
        max_k_near = max(abs(k0), abs(k1), abs(k2))
        v_tgt = (
            max(v_tgt, 4.75)
            if max_k_near < 0.28
            else track.safe_corner_speed(max(v_req, 4.75), k0, k1, k2, k3, k4)
        )
      # On circle modes, briefly straighten along tangent during pre-pop alignment
      if self.mode in ('circle_left', 'circle_right') and trick in ('wheelie', 'stoppie', 'wing_hop'):
        k0, k1, k2, k3, k4 = 0.0, 0.0, 0.0, 0.0, 0.0

      lean_lim = 0.135 if trick in ('wheelie', 'stoppie') else 0.14
      rate_lim = 0.28 if trick == 'wheelie' else (1.15 if trick == 'stoppie' else 0.65)
      lat_lim = 0.68 if trick == 'wheelie' else 0.58
      min_stop_v = 3.95 if self.mode != 'manual' else 3.55
      if trick == 'stoppie':
        speed_ok = v_fwd > min_stop_v
      elif trick == 'wheelie' and self.mode == 'figure8':
        speed_ok = v_fwd < 2.48
      else:
        speed_ok = True
      k0_lim = 0.20 if trick == 'wheelie' else 0.20
      k1_lim = 0.23 if trick == 'wheelie' else 0.24
      k2_lim = 0.24 if trick == 'wheelie' else 0.26
      k3_lim = 0.34 if trick == 'wheelie' else 0.38
      is_straight = (
          self.mode == 'figure8'
          and speed_ok
          and abs(lean) < lean_lim
          and abs(lean_rate) < rate_lim
          and abs(k0) < k0_lim
          and abs(k1) < k1_lim
          and abs(k2) < k2_lim
          and abs(k3) < k3_lim
          and abs(e_lat_raw) < lat_lim
      )
      non_fig8_lean = 0.22 if trick == 'wing_hop' else 0.12
      non_fig8_rate = 0.35 if trick == 'wing_hop' else 0.24
      is_non_fig8_ok = (
          self.mode != 'figure8'
          and abs(lean) < non_fig8_lean
          and abs(lean_rate) < non_fig8_rate
          and speed_ok
      )
      dur_pop = 1.05 if trick == 'wheelie' else (1.05 if trick == 'stoppie' else 1.35)
      if (is_straight or is_non_fig8_ok) and grounded_settled:
        self.trick_pop_at = t
        self.trick_until = t + dur_pop
      elif trick_elapsed > 7.2:
        if (
            abs(lean) < 0.28
            and abs(k1) < 0.36
            and (v_fwd > 3.15 or trick != 'stoppie')
            and grounded_settled
        ):
          self.trick_pop_at = t
          self.trick_until = t + dur_pop
        else:
          self.active_trick = ''
    pop_elapsed = (t - self.trick_pop_at) if self.trick_pop_at >= 0.0 else -1.0

    # Wheelie observation conditioning for both PPO policy and LQR fallback
    obs_target_pitch = 0.0
    obs_wheelie_active = 0.0
    if trick == 'wheelie' and pop_elapsed >= 0.0 and not self.wheelie_landed:
      pop_rem = max(0.0, self.trick_until - t)
      w_ramp = max(0.0, min(1.0, pop_elapsed / 0.18, pop_rem / 0.38))
      obs_target_pitch = -0.225 * w_ramp
      obs_wheelie_active = 1.0
    obs_no_hands = 1.0 if trick == 'no_hands' else 0.0

    # Policy evaluation at 50 Hz (every 10 physics steps)
    if self.step_counter % POLICY_SUBSTEPS == 0:
      self._ensure_policy()
      obs = policy_mod.build_obs_single(
          lean=lean,
          lean_rate=lean_rate,
          pitch=pitch,
          pitch_rate=pitch_rate,
          yaw_rate=yaw_rate,
          v_fwd=v_fwd,
          v_lat=v_lat,
          v_target=v_tgt,
          steer_q=float(data.joint(f'{p}steer_joint').qpos[0]),
          steer_qd=float(data.joint(f'{p}steer_joint').qvel[0]),
          pelvis_roll_q=float(data.joint(f'{p}pelvis_roll').qpos[0]),
          pelvis_pitch_q=float(data.joint(f'{p}pelvis_pitch').qpos[0]),
          e_lat=e_lat,
          e_psi=e_psi,
          k0=k0,
          k1=k1,
          k2=k2,
          k3=k3,
          target_pitch=obs_target_pitch,
          wheelie_active=obs_wheelie_active,
          no_hands_active=obs_no_hands,
      )
      crank_qd = float(data.joint(f'{p}crank_joint').qvel[0])
      if self.use_learned_policy and self.policy is not None:
        act = self.policy.act(obs)
        # Smooth LQR safety shield outside the MLP's validated training distribution
        ood_w = max(
            max(0.0, min(1.0, (abs(e_lat_raw) - 5.2) / 1.0)),
            max(0.0, min(1.0, (abs(lean) - 0.52) / 0.08)),
        )
        if ood_w > 0.0:
          act_lqr = self._compute_lqr_action(obs, crank_qd)
          act = (1.0 - ood_w) * act + ood_w * act_lqr
      else:
        act = self._compute_lqr_action(obs, crank_qd)

      steer_cmd = float(act[0])
      drive_cmd = max(-65.0, float(act[1]))
      if trick != 'stoppie' and v_fwd < 1.85:
        drive_cmd = max(drive_cmd, min(115.0, 50.0 * (2.05 - v_fwd)))
      pr_cmd = float(act[2])
      pp_cmd = float(act[3])
      brake_cmd = (
          max(-65.0, min(0.0, 28.0 * (v_tgt - v_fwd)))
          if (
              v_fwd > v_tgt + (0.20 if trick == 'wheelie' and self.trick_pop_at < 0.0 else 0.55)
              and trick != 'turbo_boost'
              and obs_wheelie_active < 0.5
              and front_lift < 0.02
              and rear_lift < 0.02
              and abs(lean) < 0.42
          )
          else 0.0
      )

      if trick == 'turbo_boost':
        pp_cmd = 0.16
        if v_fwd < v_tgt:
          drive_cmd = max(drive_cmd, min(155.0, 28.0 * (v_tgt - v_fwd) + 8.0))
      elif trick == 'wheelie' and pop_elapsed >= 0.0:
        pop_rem = max(0.0, self.trick_until - t)
        ramp = max(0.0, min(1.0, pop_elapsed / 0.08, pop_rem / 0.38))
        lean_safe = max(0.0, 1.0 - max(0.0, abs(lean) - 0.16) / 0.16)
        curve_safe = abs(k0) < 0.28 and (abs(k1) < 0.36 if self.mode == 'figure8' else True)
        roll_growing = abs(lean) > 0.16 or (abs(lean) > 0.08 and lean * lean_rate > 0.025)
        if not self.wheelie_landed:
          if (
              pop_elapsed < 0.50
              and v_fwd < 4.55
              and lean_safe > 0.20
              and not roll_growing
              and curve_safe
              and pitch > -0.45
          ):
            theta_des = -0.220 * ramp * lean_safe
            # Fast pop pulse while front wheel is low, then settle torque so corner speed stays safe
            pop_boost = 132.0 * max(0.0, min(1.0, (pitch + 0.14) / 0.10))
            w_drive = max(
                -45.0,
                min(
                    270.0,
                    (104.0 + pop_boost) * lean_safe
                    + 430.0 * (pitch - theta_des)
                    + 110.0 * pitch_rate,
                ),
            )
            if self.use_learned_policy:
              blend_drive = 0.20 * float(act[1]) + 0.80 * w_drive
              drive_cmd = min(blend_drive, w_drive + 5.0) if pitch < -0.06 else blend_drive
            else:
              drive_cmd = w_drive
            drive_cmd = max(-45.0, min(270.0, drive_cmd))
            brake_cmd = 0.0
          else:
            self.wheelie_landed = True
        if not self.wheelie_landed:
          pp_cmd = -0.35 * ramp
          pr_cmd = max(-0.35, min(0.35, 1.8 * lean + 0.32 * lean_rate))
          if front_lift > 0.035:
            steer_cmd = max(-0.42, min(0.42, -0.95 * lean - 0.16 * lean_rate))
          elif pop_elapsed < 0.22:
            steer_cmd = max(-0.28, min(0.28, -0.75 * lean - 0.14 * lean_rate))
        else:
          pp_cmd = 0.04
          if front_lift > 0.06:
            drive_cmd = max(-20.0, min(drive_cmd, 10.0))
            steer_cmd = max(-0.42, min(0.42, -0.95 * lean - 0.16 * lean_rate))
      elif trick == 'stoppie' and pop_elapsed >= 0.0:
        if (
            not self.stoppie_released
            and pop_elapsed < 0.55
            and v_fwd > 1.32
            and pitch < 0.34
            and abs(lean) < 0.22
        ):
          s_ramp = max(0.0, min(1.0, pop_elapsed / 0.06, (0.55 - pop_elapsed) / 0.14))
          tgt_p = 0.25 * s_ramp
          drive_cmd = 0.0
          brake_cmd = max(
              -205.0,
              min(0.0, -95.0 * s_ramp - 460.0 * (tgt_p - pitch) + 60.0 * pitch_rate),
          )
          pp_cmd = 0.34 * s_ramp
          pr_cmd = max(-0.35, min(0.35, 1.8 * lean + 0.32 * lean_rate))
          steer_cmd = max(-0.26, min(0.26, 0.45 * steer_cmd - 0.65 * lean - 0.12 * lean_rate))
        else:
          self.stoppie_released = True
          brake_cmd = 0.0
          drive_cmd = max(drive_cmd, min(155.0, 48.0 * max(0.0, v_tgt - v_fwd)))
          pp_cmd = 0.04
      elif trick == 'wing_hop' and pop_elapsed >= 0.0:
        if pop_elapsed < 0.18:
          pp_cmd = 0.16
        elif pop_elapsed < 0.46:
          pp_cmd = -0.06
        else:
          pp_cmd = 0.04
        if front_lift > 0.02 and rear_lift > 0.02:
          steer_cmd = max(
              -0.24,
              min(
                  0.24,
                  0.45 * k0 - 0.35 * e_psi - 0.55 * lean - 0.10 * lean_rate,
              ),
          )
          pr_cmd = max(-0.32, min(0.32, 1.3 * lean + 0.22 * lean_rate))
      elif trick == 'no_hands':
        pp_cmd = -0.14

      # General airborne landing stabilizer (for unexpected bumps or mid-air transitions)
      if front_lift > 0.035 and rear_lift > 0.035:
        steer_cmd = max(-0.22, min(0.22, -0.50 * lean - 0.08 * lean_rate))
        pr_cmd = max(-0.30, min(0.30, 1.1 * lean + 0.20 * lean_rate))

      self.last_steer_cmd = steer_cmd
      self.last_drive_cmd = drive_cmd
      self.last_brake_cmd = brake_cmd
      self.last_pr_cmd = pr_cmd
      self.last_pp_cmd = pp_cmd

    self.step_counter += 1

    # Wing-Flap Hop: apply balanced & roll-stabilized aerodynamic downstroke lift across inner & outer wings
    if trick == 'wing_hop' and 0.18 <= pop_elapsed < 0.42:
      wul_id = model.body(f'{p}wing_upper_left').id
      wur_id = model.body(f'{p}wing_upper_right').id
      wol_id = model.body(f'{p}wing_outer_left').id
      wor_id = model.body(f'{p}wing_outer_right').id
      qd_l = float(data.joint(f'{p}wing_l_flap').qvel[0])
      qd_r = float(data.joint(f'{p}wing_r_flap').qvel[0])
      lift_base = min(152.0, 116.0 + 15.0 * max(0.0, -0.5 * (qd_l + qd_r)))
      roll_corr = max(-0.22, min(0.22, 1.15 * lean + 0.22 * lean_rate))
      lift_l = lift_base * (1.0 - roll_corr)
      lift_r = lift_base * (1.0 + roll_corr)
      data.xfrc_applied[wul_id, 2] = 0.55 * lift_l
      data.xfrc_applied[wur_id, 2] = 0.55 * lift_r
      data.xfrc_applied[wol_id, 2] = 0.45 * lift_l
      data.xfrc_applied[wor_id, 2] = 0.45 * lift_r

    data.actuator(f'{p}steer_servo').ctrl[0] = self.last_steer_cmd
    data.actuator(f'{p}drive_motor').ctrl[0] = self.last_drive_cmd
    data.actuator(f'{p}front_brake').ctrl[0] = self.last_brake_cmd
    data.actuator(f'{p}pelvis_roll_servo').ctrl[0] = self.last_pr_cmd
    data.actuator(f'{p}pelvis_pitch_servo').ctrl[0] = self.last_pp_cmd

    # Keep pedals and pelican webbed feet horizontal throughout infinite crank rotations
    crank_q = float(data.joint(f'{p}crank_joint').qpos[0])
    crank_qd = float(data.joint(f'{p}crank_joint').qvel[0])
    cq_mid = crank_q + 0.002 * crank_qd
    data.actuator(f'{p}pedal_l_servo').ctrl[0] = -cq_mid
    data.actuator(f'{p}pedal_r_servo').ctrl[0] = -cq_mid

    pelvis_p = float(data.joint(f'{p}pelvis_pitch').qpos[0])
    data.actuator(f'{p}ankle_l_servo').ctrl[0] = -(
        float(data.joint(f'{p}hip_l_pitch').qpos[0])
        + float(data.joint(f'{p}knee_l_pitch').qpos[0])
        + pelvis_p
    )
    data.actuator(f'{p}ankle_r_servo').ctrl[0] = -(
        float(data.joint(f'{p}hip_r_pitch').qpos[0])
        + float(data.joint(f'{p}knee_r_pitch').qpos[0])
        + pelvis_p
    )

    # Neck & head look-ahead + expressive trick poses
    if self.auto_head_look:
      data.actuator(f'{p}neck_yaw_servo').ctrl[0] = 0.36 * self.last_steer_cmd
      data.actuator(f'{p}head_yaw_servo').ctrl[0] = 0.48 * self.last_steer_cmd
    else:
      data.actuator(f'{p}neck_yaw_servo').ctrl[0] = 0.0
      data.actuator(f'{p}head_yaw_servo').ctrl[0] = 0.0

    neck_pitch_cmd = -0.05 * math.sin(2.0 * crank_q)
    head_nod = 0.04 * math.sin(2.0 * crank_q)
    if trick == 'turbo_boost':
      neck_pitch_cmd = 0.26
      head_nod = -0.22
    elif trick == 'wheelie':
      neck_pitch_cmd = -0.22
      head_nod = 0.14
    elif trick == 'stoppie' and pop_elapsed >= 0.0 and not self.stoppie_released:
      neck_pitch_cmd = -0.38
      head_nod = -0.18
    if t < self.bell_nod_until:
      head_nod += 0.22 * math.sin(18.0 * (self.bell_nod_until - t))
    data.actuator(f'{p}neck_pitch_servo').ctrl[0] = neck_pitch_cmd
    data.actuator(f'{p}head_pitch_servo').ctrl[0] = head_nod

    # Beak & expandable gular pouch animation
    if t < self.beak_snap_until:
      rem = self.beak_snap_until - t
      beak_target = 0.42 * max(0.0, math.sin(14.0 * rem)) + 0.14
    elif self.beak_open_hold or trick in ('turbo_boost', 'no_hands'):
      beak_target = 0.44 + 0.05 * math.sin(4.0 * t)
    else:
      beak_target = 0.06 + 0.05 * math.sin(2.2 * t)
    data.actuator(f'{p}beak_open_servo').ctrl[0] = beak_target

    # Wing animations coordinated with tricks and user wing_mode
    if trick == 'wing_hop' and pop_elapsed >= 0.0:
      if pop_elapsed < 0.18:
        w_flap, w_yaw, w_el = 1.10, 0.35, -0.20
      elif pop_elapsed < 0.46:
        w_flap, w_yaw, w_el = -0.70, 0.35, 0.35
      else:
        w_flap, w_yaw, w_el = 0.25, 1.08, 0.05
      for side in ('l', 'r'):
        data.actuator(f'{p}wing_{side}_yaw_servo').ctrl[0] = w_yaw
        data.actuator(f'{p}wing_{side}_flap_servo').ctrl[0] = w_flap
        data.actuator(f'{p}wing_{side}_pitch_servo').ctrl[0] = -0.10
        data.actuator(f'{p}wing_{side}_elbow_servo').ctrl[0] = w_el
    elif trick == 'turbo_boost':
      if trick_elapsed < 0.55:
        phase = 11.0 * trick_elapsed
        for side in ('l', 'r'):
          data.actuator(f'{p}wing_{side}_yaw_servo').ctrl[0] = 1.15
          data.actuator(f'{p}wing_{side}_flap_servo').ctrl[0] = 0.20 + 0.70 * math.sin(phase)
          data.actuator(f'{p}wing_{side}_pitch_servo').ctrl[0] = -0.18
          data.actuator(f'{p}wing_{side}_elbow_servo').ctrl[0] = -0.45 * math.cos(phase)
      else:
        for side in ('l', 'r'):
          data.actuator(f'{p}wing_{side}_yaw_servo').ctrl[0] = 1.28
          data.actuator(f'{p}wing_{side}_flap_servo').ctrl[0] = 0.06
          data.actuator(f'{p}wing_{side}_pitch_servo').ctrl[0] = -0.22
          data.actuator(f'{p}wing_{side}_elbow_servo').ctrl[0] = 0.10
    elif trick in ('wheelie', 'stoppie', 'no_hands'):
      phase = 6.5 * trick_elapsed
      for side in ('l', 'r'):
        data.actuator(f'{p}wing_{side}_yaw_servo').ctrl[0] = 1.10
        data.actuator(f'{p}wing_{side}_flap_servo').ctrl[0] = (
            0.40 + 0.30 * math.sin(phase) if trick == 'no_hands' else 0.30 - 0.35 * (lean if side == 'l' else -lean)
        )
        data.actuator(f'{p}wing_{side}_pitch_servo').ctrl[0] = -0.10
        data.actuator(f'{p}wing_{side}_elbow_servo').ctrl[0] = 0.05
    elif trick == 's_carve':
      data.actuator(f'{p}wing_l_yaw_servo').ctrl[0] = 1.05
      data.actuator(f'{p}wing_r_yaw_servo').ctrl[0] = 1.05
      data.actuator(f'{p}wing_l_flap_servo').ctrl[0] = 0.24 - 0.65 * lean
      data.actuator(f'{p}wing_r_flap_servo').ctrl[0] = 0.24 + 0.65 * lean
      data.actuator(f'{p}wing_l_pitch_servo').ctrl[0] = -0.08
      data.actuator(f'{p}wing_r_pitch_servo').ctrl[0] = -0.08
      data.actuator(f'{p}wing_l_elbow_servo').ctrl[0] = 0.05
      data.actuator(f'{p}wing_r_elbow_servo').ctrl[0] = 0.05
    elif self.wing_mode == 'flap' or t < self.flap_burst_until:
      amp = self.flap_intensity if self.wing_mode == 'flap' else 0.92
      phase = 7.2 * t
      flap = 0.18 + 0.72 * amp * math.sin(phase)
      elbow = -0.52 * amp * math.cos(phase)
      for side in ('l', 'r'):
        data.actuator(f'{p}wing_{side}_yaw_servo').ctrl[0] = 1.05
        data.actuator(f'{p}wing_{side}_flap_servo').ctrl[0] = flap
        data.actuator(f'{p}wing_{side}_pitch_servo').ctrl[0] = -0.15 * amp * math.sin(phase)
        data.actuator(f'{p}wing_{side}_elbow_servo').ctrl[0] = elbow
    elif self.wing_mode == 'glide':
      data.actuator(f'{p}wing_l_yaw_servo').ctrl[0] = 1.12
      data.actuator(f'{p}wing_r_yaw_servo').ctrl[0] = 1.12
      data.actuator(f'{p}wing_l_flap_servo').ctrl[0] = 0.22 - 0.65 * lean
      data.actuator(f'{p}wing_r_flap_servo').ctrl[0] = 0.22 + 0.65 * lean
      data.actuator(f'{p}wing_l_pitch_servo').ctrl[0] = -0.08
      data.actuator(f'{p}wing_r_pitch_servo').ctrl[0] = -0.08
      data.actuator(f'{p}wing_l_elbow_servo').ctrl[0] = 0.05
      data.actuator(f'{p}wing_r_elbow_servo').ctrl[0] = 0.05
    else:
      steer_q = float(data.joint(f'{p}steer_joint').qpos[0])
      data.actuator(f'{p}wing_l_yaw_servo').ctrl[0] = -0.22 * steer_q
      data.actuator(f'{p}wing_r_yaw_servo').ctrl[0] = 0.22 * steer_q
      data.actuator(f'{p}wing_l_flap_servo').ctrl[0] = 0.04 * math.sin(2.0 * crank_q)
      data.actuator(f'{p}wing_r_flap_servo').ctrl[0] = -0.04 * math.sin(2.0 * crank_q)
      data.actuator(f'{p}wing_l_pitch_servo').ctrl[0] = 0.16 * steer_q
      data.actuator(f'{p}wing_r_pitch_servo').ctrl[0] = -0.16 * steer_q
      data.actuator(f'{p}wing_l_elbow_servo').ctrl[0] = 0.0
      data.actuator(f'{p}wing_r_elbow_servo').ctrl[0] = 0.0

  @messages.handler
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    data = event.data
    p = BIKE_PREFIX
    pos = data.body(f'{p}pelican_bicycle').xpos
    px, py = float(pos[0]), float(pos[1])
    if self.prev_xy_valid:
      step_dist = math.hypot(px - self.prev_x, py - self.prev_y)
      if step_dist < 0.5:
        self.distance_m += step_dist
    self.prev_x = px
    self.prev_y = py
    self.prev_xy_valid = True

    self.telemetry_counter += 1
    if self.telemetry_counter % 10 != 0:
      return

    xmat = data.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
    fwd = xmat[:, 0]
    up = xmat[:, 2]
    yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    left_h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    lean = math.atan2(-float(np.dot(up, left_h)), max(1e-4, float(up[2])))
    pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))

    v_fwd = float(data.sensor(f'{p}bike_vel').data[0])
    steer = float(data.joint(f'{p}steer_joint').qpos[0])
    omega_crank = float(data.joint(f'{p}crank_joint').qvel[0])
    beak_q = float(data.joint(f'{p}beak_open').qpos[0])

    front_lift = float(data.body(f'{p}front_wheel').xpos[2]) - WHEEL_RADIUS
    rear_lift = float(data.body(f'{p}rear_wheel').xpos[2]) - WHEEL_RADIUS
    airborne = front_lift > 0.04 and rear_lift > 0.04

    if self.use_learned_policy and self.policy is not None:
      it_n = int(self.policy.metadata.get('iter', 0))
      err_n = self.policy.metadata.get('eval_err_m', 0.0)
      pol_lbl = f'Trained PPO MLP (iter {it_n}, eval {err_n:.2f}m)'
    else:
      pol_lbl = 'LQR Baseline'

    self.handle.send_to_viewer(
        user_messages.RideTelemetry(
            sim_time=float(data.time),
            speed_mps=v_fwd,
            target_speed_mps=self.target_speed,
            lean_deg=math.degrees(lean),
            pitch_deg=-math.degrees(pitch),  # +deg = nose-up wheelie
            steer_deg=math.degrees(steer),
            crank_rpm=(omega_crank * 60.0) / (2.0 * math.pi),
            distance_m=self.distance_m,
            road_err_m=self.road_err_m,
            pos_x=px,
            pos_y=py,
            heading_deg=math.degrees(yaw),
            upright=(abs(lean) < 1.00 and float(up[2]) > 0.45),
            airborne=airborne,
            beak_deg=math.degrees(beak_q),
            mode=self.mode,
            active_trick=TRICK_LABELS.get(self.active_trick, ''),
            pending_trick=TRICK_LABELS.get(self.pending_trick, ''),
            trick_count=self.trick_count,
            policy_label=pol_lbl,
        )
    )
