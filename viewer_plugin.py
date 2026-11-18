"""Viewer-side Dear ImGui & ImPlot Studio plugin for the Pelican Bicycle ride."""

from __future__ import annotations

import collections
import os
import pathlib

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins
from mujoco.studio.implot import implot

imgui = app_plugins.imgui
ux = app_plugins.ux

WORKSPACE_DIR = pathlib.Path(os.path.abspath(__file__)).parent

RIDE_MODES = ('figure8', 'circle_left', 'circle_right', 'cruise', 'manual')
RIDE_MODE_LABELS = [
    'Figure-8 Track',
    'Circle Left',
    'Circle Right',
    'Straight Cruise',
    'Manual Steer',
]

WING_MODES = ('handlebars', 'glide', 'flap')
WING_MODE_LABELS = [
    'Hold Handlebars',
    'Spread Glide',
    'Flap Wings',
]

ENV_MODES = ('coastal_road', 'slalom_cones', 'checker')
ENV_MODE_LABELS = [
    'Coastal Boardwalk',
    'Slalom Cones',
    'Studio Checker',
]


def _short_path(path: pathlib.Path) -> str:
  home = pathlib.Path.home()
  return (
      '~/' + str(path.relative_to(home))
      if path.is_relative_to(home)
      else str(path)
  )


class ViewerPlugin(plugin.ViewerPlugin):
  """Pelican Bicycle Ride Controls, Trick Buttons, Camera Follow, and Telemetry."""

  # State persisted as class-level defaults for seamless hot-reload
  mode_idx: int = 0
  target_speed: float = 3.4
  manual_steer: float = 0.0
  wing_mode_idx: int = 0
  flap_intensity: float = 0.75
  beak_open_hold: bool = False
  auto_head_look: bool = True
  env_idx: int = 0
  follow_camera: bool = True
  use_learned_policy: bool = True
  camera_needs_sync: bool = True
  window_open: bool = True

  telemetry: user_messages.RideTelemetry = user_messages.RideTelemetry()
  hist_t: collections.deque = collections.deque(maxlen=240)
  hist_lean: collections.deque = collections.deque(maxlen=240)
  hist_pitch: collections.deque = collections.deque(maxlen=240)
  hist_speed: collections.deque = collections.deque(maxlen=240)
  last_hist_time: float = -1.0

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    del event
    self.camera_needs_sync = True
    self.hist_t.clear()
    self.hist_lean.clear()
    self.hist_pitch.clear()
    self.hist_speed.clear()
    self.last_hist_time = -1.0

  @messages.handler
  def on_telemetry(self, snapshot: user_messages.RideTelemetry) -> None:
    self.telemetry = snapshot
    if snapshot.sim_time < self.last_hist_time:
      self.hist_t.clear()
      self.hist_lean.clear()
      self.hist_pitch.clear()
      self.hist_speed.clear()
      self.last_hist_time = -1.0
    if snapshot.sim_time - self.last_hist_time >= 0.04:
      self.hist_t.append(snapshot.sim_time)
      self.hist_lean.append(snapshot.lean_deg)
      self.hist_pitch.append(getattr(snapshot, 'pitch_deg', 0.0))
      self.hist_speed.append(snapshot.speed_mps)
      self.last_hist_time = snapshot.sim_time

  @messages.handler
  def on_update(self, event: messages.UpdateEvent) -> None:
    del event
    if self.camera_needs_sync:
      self.camera_needs_sync = False
      self._apply_follow_camera(reset_angles=True)

  def _apply_follow_camera(self, *, reset_angles: bool = False) -> None:
    if self.viewer is None or self.viewer.model is None:
      return
    cam = self.viewer.camera
    if self.follow_camera:
      try:
        body_id = self.viewer.model.body('pb_pelican_bicycle').id
        cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
        cam.trackbodyid = body_id
        cam.fixedcamid = -1
        if reset_angles:
          cam.distance = 3.2
          cam.azimuth = 135.0
          cam.elevation = -17.0
      except KeyError:
        pass
    else:
      if cam.type == int(mujoco.mjtCamera.mjCAMERA_TRACKING):
        cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        cam.trackbodyid = -1

  def _set_fixed_camera(self, cam_name: str) -> None:
    if self.viewer is None or self.viewer.model is None:
      return
    try:
      cam_id = self.viewer.model.camera(cam_name).id
      self.follow_camera = False
      ux.set_camera(self.viewer.model, self.viewer.camera, cam_id)
    except KeyError:
      pass

  def _send_config(self) -> None:
    if self.viewer is None:
      return
    self.viewer.send_to_sim(
        user_messages.RideConfig(
            mode=RIDE_MODES[self.mode_idx],
            target_speed=self.target_speed,
            manual_steer=self.manual_steer,
            wing_mode=WING_MODES[self.wing_mode_idx],
            flap_intensity=self.flap_intensity,
            beak_open_hold=self.beak_open_hold,
            auto_head_look=self.auto_head_look,
            environment=ENV_MODES[self.env_idx],
            follow_camera=self.follow_camera,
            use_learned_policy=self.use_learned_policy,
        )
    )

  def _send_action(self, action: str) -> None:
    if self.viewer is None:
      return
    self.viewer.send_to_sim(user_messages.ActionEvent(action=action))

  def build_gui(self) -> None:
    t = self.telemetry
    policy_label = getattr(t, 'policy_label', 'Trained PPO MLP')
    imgui.TextUnformatted(f'Workspace: {_short_path(WORKSPACE_DIR)}')
    imgui.SetItemTooltip(
        'Hot-reloading Python scene.\nEdit sim_plugin.py or viewer_plugin.py'
    )
    imgui.TextUnformatted(f'Controller: {policy_label}')
    imgui.Separator()

    table_flags = (
        int(imgui.TableFlags.SizingStretchProp)
        | int(imgui.TableFlags.NoPadOuterX)
    )
    changed = False
    if imgui.BeginTable('##pelican_controls_tbl', 2, table_flags):
      imgui.TableSetupColumn('Label', int(imgui.TableColumnFlags.WidthFixed))
      imgui.TableSetupColumn('Widget', int(imgui.TableColumnFlags.WidthStretch))

      imgui.TableNextRow()
      imgui.TableNextColumn()
      imgui.AlignTextToFramePadding()
      imgui.TextUnformatted('Ride Path:')
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      c, self.mode_idx = imgui.Combo(
          '##ride_mode', self.mode_idx, RIDE_MODE_LABELS
      )
      changed = changed or c

      imgui.TableNextRow()
      imgui.TableNextColumn()
      imgui.AlignTextToFramePadding()
      imgui.TextUnformatted('Target Speed:')
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      c, self.target_speed = imgui.SliderFloat(
          '##speed', self.target_speed, 1.8, 6.0, '%.2f m/s'
      )
      changed = changed or c

      if RIDE_MODES[self.mode_idx] == 'manual':
        imgui.TableNextRow()
        imgui.TableNextColumn()
        imgui.AlignTextToFramePadding()
        imgui.TextUnformatted('Manual Steer:')
        imgui.TableNextColumn()
        imgui.SetNextItemWidth(-1)
        c, self.manual_steer = imgui.SliderFloat(
            '##manual_steer', self.manual_steer, -1.0, 1.0, '%.2f'
        )
        changed = changed or c

      imgui.TableNextRow()
      imgui.TableNextColumn()
      imgui.AlignTextToFramePadding()
      imgui.TextUnformatted('Wing Pose:')
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      c, self.wing_mode_idx = imgui.Combo(
          '##wing_mode', self.wing_mode_idx, WING_MODE_LABELS
      )
      changed = changed or c

      if WING_MODES[self.wing_mode_idx] == 'flap':
        imgui.TableNextRow()
        imgui.TableNextColumn()
        imgui.AlignTextToFramePadding()
        imgui.TextUnformatted('Flap Power:')
        imgui.TableNextColumn()
        imgui.SetNextItemWidth(-1)
        c, self.flap_intensity = imgui.SliderFloat(
            '##flap_amp', self.flap_intensity, 0.2, 1.2, '%.2f'
        )
        changed = changed or c

      imgui.TableNextRow()
      imgui.TableNextColumn()
      imgui.AlignTextToFramePadding()
      imgui.TextUnformatted('Environment:')
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      c, self.env_idx = imgui.Combo(
          '##env_mode', self.env_idx, ENV_MODE_LABELS
      )
      changed = changed or c

      imgui.EndTable()

    c_pol, self.use_learned_policy = imgui.Checkbox(
        'Use Trained Policy (PPO MLP)', self.use_learned_policy
    )
    imgui.SetItemTooltip(
        'Run the trained neural policy.\nUncheck to compare LQR baseline.'
    )
    c_beak, self.beak_open_hold = imgui.Checkbox(
        'Hold Beak & Pouch Open', self.beak_open_hold
    )
    c_head, self.auto_head_look = imgui.Checkbox(
        'Look Into Turns', self.auto_head_look
    )
    c_cam, self.follow_camera = imgui.Checkbox(
        'Chase Camera Follow', self.follow_camera
    )
    if c_cam:
      self._apply_follow_camera(reset_angles=self.follow_camera)

    if changed or c_pol or c_beak or c_head or c_cam:
      self._send_config()

    grid_flags = (
        int(imgui.TableFlags.SizingStretchSame)
        | int(imgui.TableFlags.NoPadOuterX)
    )
    imgui.Separator()
    imgui.TextUnformatted('Bike & Pelican Tricks')
    if imgui.BeginTable('##tricks_grid', 3, grid_flags):
      imgui.TableSetupColumn('C1', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('C2', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('C3', int(imgui.TableColumnFlags.WidthStretch))

      imgui.TableNextRow()
      imgui.TableNextColumn()
      if imgui.Button('Turbo Boost!###BoostBtn', imgui.Vec2(-1, 0)):
        self._send_action('turbo_boost')
      imgui.SetItemTooltip(
          'Surge to 5.8 m/s in aero tuck\nand carve along the road!'
      )
      imgui.TableNextColumn()
      if imgui.Button('Wheelie!###WheelieBtn', imgui.Vec2(-1, 0)):
        self._send_action('wheelie')
      imgui.SetItemTooltip(
          'Pop the front wheel up,\nbalance, and touch down cleanly.'
      )
      imgui.TableNextColumn()
      if imgui.Button('Wing-Flap Hop!###HopBtn', imgui.Vec2(-1, 0)):
        self._send_action('wing_hop')
      imgui.SetItemTooltip(
          'Power-flap wings to launch\nboth wheels off the road!'
      )

      imgui.TableNextRow()
      imgui.TableNextColumn()
      if imgui.Button('Stoppie!###StoppieBtn', imgui.Vec2(-1, 0)):
        self._send_action('stoppie')
      imgui.SetItemTooltip(
          'Apply modulated front brake\nfor a nose-wheelie balance.'
      )
      imgui.TableNextColumn()
      if imgui.Button('No-Hands!###NoHandsBtn', imgui.Vec2(-1, 0)):
        self._send_action('no_hands')
      imgui.SetItemTooltip(
          'Spread wings hands-free while\ncarving along the road.'
      )
      imgui.TableNextColumn()
      if imgui.Button('S-Carve!###CarveBtn', imgui.Vec2(-1, 0)):
        self._send_action('s_carve')
      imgui.SetItemTooltip(
          'Weave rhythmic S-turns across\nthe lane and re-center.'
      )

      imgui.EndTable()

    imgui.Separator()
    imgui.TextUnformatted('Pelican Actions & Disturbance Test')
    if imgui.BeginTable('##actions_grid', 3, grid_flags):
      imgui.TableSetupColumn('A1', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('A2', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('A3', int(imgui.TableColumnFlags.WidthStretch))

      imgui.TableNextRow()
      imgui.TableNextColumn()
      if imgui.Button('Flap Wings!###FlapBtn', imgui.Vec2(-1, 0)):
        self._send_action('flap_burst')
      imgui.TableNextColumn()
      if imgui.Button('Snap Beak!###SnapBtn', imgui.Vec2(-1, 0)):
        self._send_action('snap_beak')
      imgui.TableNextColumn()
      if imgui.Button('Nod & Bell###BellBtn', imgui.Vec2(-1, 0)):
        self._send_action('ring_bell')

      imgui.TableNextRow()
      imgui.TableNextColumn()
      if imgui.Button('Push Left###PushLBtn', imgui.Vec2(-1, 0)):
        self._send_action('nudge_left')
      imgui.SetItemTooltip(
          'Apply a lateral impulse\nto test counter-steering recovery.'
      )
      imgui.TableNextColumn()
      if imgui.Button('Push Right###PushRBtn', imgui.Vec2(-1, 0)):
        self._send_action('nudge_right')
      imgui.SetItemTooltip(
          'Apply a lateral impulse\nto test counter-steering recovery.'
      )
      imgui.TableNextColumn()
      if imgui.Button('Reset Ride###ResetRideBtn', imgui.Vec2(-1, 0)):
        self._send_action('reset_ride')

      imgui.EndTable()

    imgui.Separator()
    imgui.TextUnformatted('Camera Views:')
    if imgui.BeginTable('##cameras_grid', 4, table_flags):
      imgui.TableSetupColumn('K1', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('K2', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('K3', int(imgui.TableColumnFlags.WidthStretch))
      imgui.TableSetupColumn('K4', int(imgui.TableColumnFlags.WidthStretch))

      imgui.TableNextRow()
      imgui.TableNextColumn()
      if imgui.Button('Orbit###CamOrbitBtn', imgui.Vec2(-1, 0)):
        self.follow_camera = True
        self._apply_follow_camera(reset_angles=True)
      imgui.SetItemTooltip(
          'Follow orbit camera behind\nthe pelican and bicycle.'
      )
      imgui.TableNextColumn()
      if imgui.Button('Side###CamSideBtn', imgui.Vec2(-1, 0)):
        self._set_fixed_camera('pb_side_cam')
      imgui.TableNextColumn()
      if imgui.Button('Beak Cam###CamBeakBtn', imgui.Vec2(-1, 0)):
        self._set_fixed_camera('pb_beak_cam')
      imgui.TableNextColumn()
      if imgui.Button('Chase###CamChaseBtn', imgui.Vec2(-1, 0)):
        self._set_fixed_camera('pb_chase_cam')

      imgui.EndTable()

    imgui.Separator()
    airborne = getattr(t, 'airborne', False)
    active_trick = getattr(t, 'active_trick', '')
    pending_trick = getattr(t, 'pending_trick', '')
    trick_count = getattr(t, 'trick_count', 0)
    road_err = getattr(t, 'road_err_m', 0.0)
    pitch_deg = getattr(t, 'pitch_deg', 0.0)

    if not t.upright:
      status_str = 'TIPPED (click Reset Ride)'
    elif airborne:
      status_str = 'AIRBORNE!'
    else:
      status_str = 'UPRIGHT'
    trick_str = f' | Trick: {active_trick}' if active_trick else ''
    queue_str = f' | Next: {pending_trick}' if pending_trick else ''
    imgui.TextUnformatted(
        f'Status: {status_str}{trick_str}{queue_str} | Tricks: {trick_count}'
    )
    imgui.TextUnformatted(
        f't = {t.sim_time:.1f} s | Dist = {t.distance_m:.1f} m | Road Err = {road_err:.2f} m'
    )
    imgui.TextUnformatted(
        f'Speed: {t.speed_mps:.2f} m/s | Cadence: {t.crank_rpm:.0f} rpm'
    )
    imgui.TextUnformatted(
        f'Lean: {t.lean_deg:+.1f} deg | Pitch: {pitch_deg:+.1f} deg | Steer: {t.steer_deg:+.1f} deg'
    )

    if len(self.hist_t) >= 4:
      ts = list(self.hist_t)
      leans = list(self.hist_lean)
      pitches = list(self.hist_pitch)
      speeds4 = [4.0 * s for s in self.hist_speed]
      peak_y = max(
          max((abs(v) for v in leans), default=0.0),
          max((abs(v) for v in pitches), default=0.0),
          max((abs(v) for v in speeds4), default=0.0),
      )
      y_lim = max(35.0, min(110.0, peak_y + 6.0))
      if implot.BeginPlot(
          'Lean, Pitch & Speed##ride_plot', imgui.Vec2(-1, 150)
      ):
        implot.SetupAxesLimits(
            ts[0], max(ts[-1], ts[0] + 1.0), -y_lim, y_lim, int(implot.Cond.Always)
        )
        implot.PlotLine('Lean (deg)', ts, leans)
        implot.PlotLine('Pitch (deg)', ts, pitches)
        implot.PlotLine('Speed x4', ts, speeds4)
        implot.EndPlot()

    imgui.Separator()
    if imgui.Button('Bake to scene.xml###BakeBtn'):
      self._send_action('bake_xml')
    imgui.SetItemTooltip(
        'Write the current scene to\nscene.xml in the workspace.'
    )

  @messages.handler
  def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
    del event
    if not self.window_open:
      return
    expanded, self.window_open = imgui.Begin(
        'Pelican Bicycle Controls###PelicanBikeWindow', self.window_open
    )
    try:
      if not expanded:
        return
      self.build_gui()
    finally:
      imgui.End()
