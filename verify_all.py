"""Comprehensive verification suite for Pelican Bike policy, tricks, and modes."""

from __future__ import annotations

import math
import numpy as np
import mujoco
import messages as user_messages
import sim_plugin
import track


class DummyHandle:
  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self.model = model
    self.data = data
    self.last_telemetry: user_messages.RideTelemetry | None = None

  def send_to_viewer(self, msg) -> None:
    if isinstance(msg, user_messages.RideTelemetry):
      self.last_telemetry = msg

  def set_model(self, *args, **kwargs) -> None:
    pass

  def reload(self) -> None:
    pass

  def dispatch(self, *args, **kwargs) -> None:
    pass


class DummyEvent:
  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self.model = model
    self.data = data


def make_sim(env: str = 'coastal_road', mode: str = 'figure8', speed: float = 3.4):
  sp = sim_plugin.SimPlugin()
  sp.environment = env
  sp.mode = mode
  sp.target_speed = speed
  sp.use_learned_policy = True
  spec = sp.build_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  sp.handle = DummyHandle(model, data)
  sp._ensure_policy(force=True)
  sp._initialize_rolling_state(model, data)
  return sp, model, data


def run_steps(sp, model, data, seconds: float):
  n_steps = int(round(seconds / model.opt.timestep))
  ev = DummyEvent(model, data)
  errs = []
  speeds = []
  max_lean = 0.0
  min_up_z = 1.0
  max_front_lift = 0.0
  max_rear_lift = 0.0
  max_both_lift = 0.0
  max_wing_yaw = 0.0
  p = sim_plugin.BIKE_PREFIX

  for i in range(n_steps):
    sp.on_pre_step(ev)
    mujoco.mj_step(model, data)
    sp.on_post_step(ev)
    if i % 10 == 0:
      xmat = data.body(f'{p}pelican_bicycle').xmat.reshape(3, 3)
      up_z = float(xmat[2, 2])
      min_up_z = min(min_up_z, up_z)
      fwd = xmat[:, 0]
      up = xmat[:, 2]
      yaw = math.atan2(float(fwd[1]), float(fwd[0]))
      lh = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
      lean = abs(math.degrees(math.atan2(-float(np.dot(up, lh)), max(1e-4, up_z))))
      max_lean = max(max_lean, lean)
      errs.append(sp.road_err_m)
      speeds.append(float(data.sensor(f'{p}bike_vel').data[0]))
      fl = float(data.body(f'{p}front_wheel').xpos[2]) - sim_plugin.WHEEL_RADIUS
      rl = float(data.body(f'{p}rear_wheel').xpos[2]) - sim_plugin.WHEEL_RADIUS
      max_front_lift = max(max_front_lift, fl)
      max_rear_lift = max(max_rear_lift, rl)
      max_both_lift = max(max_both_lift, min(fl, rl))
      max_wing_yaw = max(max_wing_yaw, abs(float(data.joint(f'{p}wing_l_yaw').qpos[0])))
      if lean > 58.0 or up_z < 0.45:
        return {
            'upright': False,
            'fell_at': float(data.time),
            'mean_err': float(np.mean(errs)),
            'max_err': float(np.max(errs)),
            'max_lean': max_lean,
            'max_speed': float(np.max(speeds)),
            'min_speed': float(np.min(speeds)),
            'dist': sp.distance_m,
            'max_front_lift_cm': max_front_lift * 100.0,
            'max_rear_lift_cm': max_rear_lift * 100.0,
            'max_both_lift_cm': max_both_lift * 100.0,
            'max_wing_yaw': max_wing_yaw,
        }
  return {
      'upright': True,
      'fell_at': None,
      'mean_err': float(np.mean(errs)),
      'max_err': float(np.max(errs)),
      'max_lean': max_lean,
      'max_speed': float(np.max(speeds)),
      'min_speed': float(np.min(speeds)),
      'dist': sp.distance_m,
      'max_front_lift_cm': max_front_lift * 100.0,
      'max_rear_lift_cm': max_rear_lift * 100.0,
      'max_both_lift_cm': max_both_lift * 100.0,
      'max_wing_yaw': max_wing_yaw,
  }


def main() -> None:
  # 1. 180-second endurance test (same time as user screenshot t=177.1s!)
  sp, m, d = make_sim('coastal_road', 'figure8', 3.4)
  assert sp.policy is not None, 'trained_policy.npz failed to load!'
  r180 = run_steps(sp, m, d, 180.0)
  print('1. 180s Fig8 Endurance:', r180)
  assert r180['upright'], f'Fell at {r180["fell_at"]}'
  assert r180['dist'] > 500.0, f'Too short distance: {r180["dist"]}'
  assert r180['mean_err'] < 0.30, f'High mean road error: {r180["mean_err"]}'
  assert r180['max_err'] < 0.88, f'Left asphalt road: {r180["max_err"]}'

  # 2. All 6 Tricks triggered sequentially while biking continuously along Figure-8!
  sp, m, d = make_sim('coastal_road', 'figure8', 3.4)
  run_steps(sp, m, d, 0.5)
  tricks_expected = [
      ('turbo_boost', 6.5, lambda res: res['max_speed'] > 5.0 and res['mean_err'] < 0.65),
      ('wheelie', 6.8, lambda res: res['max_front_lift_cm'] > 18.0 and res['mean_err'] < 0.65),
      ('wing_hop', 6.8, lambda res: res['max_both_lift_cm'] > 10.0 and res['mean_err'] < 0.65),
      ('stoppie', 7.5, lambda res: res['max_rear_lift_cm'] > 16.0 and res['mean_err'] < 0.65 and res['min_speed'] > 1.0),
      ('no_hands', 4.5, lambda res: res['max_wing_yaw'] > 0.8 and res['mean_err'] < 0.65 and res['min_speed'] > 1.5),
      ('s_carve', 5.0, lambda res: res['max_err'] > 0.35 and res['mean_err'] < 0.65 and res['min_speed'] > 1.5),
  ]
  for act, dur, check_fn in tricks_expected:
    sp.on_action(user_messages.ActionEvent(action=act))
    res = run_steps(sp, m, d, dur)
    print(f'2. Trick {act:12s}:', res)
    assert res['upright'], f'Trick {act} tipped at {res["fell_at"]}!'
    assert check_fn(res), f'Trick {act} failed physical signature check: {res}'
  assert sp.trick_count >= 6, f'Expected trick_count >= 6, got {sp.trick_count}'

  # 3. Straight High-Speed Cruise (5.8 m/s & 6.0 m/s) + Cruise Turbo Boost (zero weave)
  for spd in (5.8, 6.0):
    sp, m, d = make_sim('checker', 'cruise', spd)
    rc = run_steps(sp, m, d, 15.0)
    print(f'3. Cruise {spd} m/s:', rc)
    assert rc['upright'], f'Cruise {spd} fell!'
    assert rc['max_lean'] < 5.0, f'High-speed weave detected at {spd}: {rc["max_lean"]}'
    assert rc['mean_err'] < 0.15, f'High cruise err at {spd}: {rc["mean_err"]}'
  sp, m, d = make_sim('checker', 'cruise', 3.4)
  run_steps(sp, m, d, 1.5)
  sp.on_action(user_messages.ActionEvent(action='turbo_boost'))
  rcb = run_steps(sp, m, d, 6.5)
  print('3b. Cruise + turbo_boost:', rcb)
  assert rcb['upright'] and rcb['max_speed'] > 5.5 and rcb['max_lean'] < 5.0

  # 4. Circle Left / Right at 3.4-5.5 m/s + Multi-lap Sequential 6 Tricks
  for mode in ('circle_left', 'circle_right'):
    sp, m, d = make_sim('checker', mode, 5.5)
    r_fast = run_steps(sp, m, d, 15.0)
    print(f'4. {mode} 5.5 m/s:', r_fast)
    assert r_fast['upright'], f'{mode} 5.5 fell!'
    sp, m, d = make_sim('checker', mode, 3.6)
    run_steps(sp, m, d, 5.0)
    tc0 = sp.trick_count
    for act in ('turbo_boost', 'wheelie', 'wing_hop', 'stoppie', 'no_hands', 's_carve'):
      sp.on_action(user_messages.ActionEvent(action=act))
      rt = run_steps(sp, m, d, 7.0)
      assert rt['upright'], f'{mode} multi-lap + {act} fell!'
    print(f'4b. Multi-lap {mode} + 6 sequential tricks: PASS! tricks={sp.trick_count - tc0} final_err={sp.road_err_m:.2f}m')
    assert sp.trick_count - tc0 >= 6, f'{mode} only executed {sp.trick_count - tc0} tricks!'

  # 5. Manual Turning Mode (+0.7) + All Tricks
  for act in ('turbo_boost', 'wheelie', 'wing_hop', 'stoppie', 'no_hands', 's_carve'):
    sp, m, d = make_sim('checker', 'manual', 3.4)
    sp.manual_steer = 0.7
    run_steps(sp, m, d, 1.5)
    tc0 = sp.trick_count
    sp.on_action(user_messages.ActionEvent(action=act))
    rt = run_steps(sp, m, d, 6.5)
    print(f'5. manual(+0.7) + {act:12s}: upright={rt["upright"]} tc={sp.trick_count-tc0} fl={rt["max_front_lift_cm"]:.1f} rl={rt["max_rear_lift_cm"]:.1f}')
    assert rt['upright'], f'manual + {act} fell!'
    assert sp.trick_count - tc0 >= 1, f'manual + {act} did not execute!'

  # 6. Mid-Air & Active-Trick Interruptions
  for first, delay, second in (
      ('wing_hop', 0.28, 'stoppie'),
      ('wing_hop', 0.28, 'wheelie'),
      ('wing_hop', 0.28, 'wing_hop'),
      ('stoppie', 0.20, 'turbo_boost'),
      ('stoppie', 0.20, 'wheelie'),
      ('wheelie', 0.30, 'stoppie'),
      ('wheelie', 0.30, 'wing_hop'),
  ):
    sp, m, d = make_sim('coastal_road', 'figure8', 3.4)
    run_steps(sp, m, d, 0.5)
    sp.on_action(user_messages.ActionEvent(action=first))
    ev = DummyEvent(m, d)
    popped = False
    t_pop = 0.0
    for _ in range(int(7.0 / m.opt.timestep)):
      sp.on_pre_step(ev)
      mujoco.mj_step(m, d)
      sp.on_post_step(ev)
      if not popped and sp.trick_pop_at >= 0.0:
        popped = True
        t_pop = float(d.time)
      if popped and float(d.time) >= t_pop + delay:
        sp.on_action(user_messages.ActionEvent(action=second))
        break
    ri = run_steps(sp, m, d, 7.5)
    print(f'6. Interrupt {first}->{second}: upright={ri["upright"]} tc={sp.trick_count} max_lean={ri["max_lean"]:.1f}')
    assert ri['upright'], f'Interrupt {first}->{second} fell!'
    assert sp.trick_count >= 2, f'Interrupt {first}->{second} dropped trick count: {sp.trick_count}'

  # 7. Off-Track Recovery (far offsets & opposite heading)
  for sx, sy, syaw in ((0.0, 8.0, math.pi), (15.0, 0.0, 0.0), (-12.0, 6.0, 0.5 * math.pi)):
    sp, m, d = make_sim('checker', 'figure8', 3.4)
    d.joint('pb_free').qpos[0:3] = [sx, sy, 0.0]
    d.joint('pb_free').qpos[3:7] = [math.cos(0.5 * syaw), 0.0, 0.0, math.sin(0.5 * syaw)]
    d.joint('pb_free').qvel[0:3] = [3.4 * math.cos(syaw), 3.4 * math.sin(syaw), 0.0]
    mujoco.mj_forward(m, d)
    sp.track_s_idx = None
    ro = run_steps(sp, m, d, 18.0)
    print(f'7. Off-track ({sx},{sy}): upright={ro["upright"]} final_err={sp.road_err_m:.3f}m')
    assert ro['upright'] and sp.road_err_m < 0.65, f'Off-track ({sx},{sy}) failed: {ro}, final={sp.road_err_m}'

  # 8. Disturbance pushes (Push Left & Push Right + Flap + Snap + Bell)
  sp, m, d = make_sim('coastal_road', 'figure8', 3.4)
  run_steps(sp, m, d, 2.0)
  for act in ('nudge_left', 'flap_burst', 'snap_beak', 'ring_bell'):
    sp.on_action(user_messages.ActionEvent(action=act))
  res_l = run_steps(sp, m, d, 6.0)
  assert res_l['upright'], f'Push Left failed: {res_l}'
  sp.on_action(user_messages.ActionEvent(action='nudge_right'))
  res_r = run_steps(sp, m, d, 6.0)
  assert res_r['upright'], f'Push Right failed: {res_r}'
  print('8. Push Left/Right + Pelican actions: PASS', res_l['max_err'], res_r['max_err'])

  # 9. LQR Baseline + Slalom Cones + Reset
  sp, m, d = make_sim('checker', 'cruise', 5.8)
  sp.use_learned_policy = False
  rlq = run_steps(sp, m, d, 12.0)
  assert rlq['upright'] and rlq['max_lean'] < 5.0, f'LQR cruise 5.8 failed: {rlq}'

  sp, m, d = make_sim('coastal_road', 'figure8', 3.4)
  sp.use_learned_policy = False
  run_steps(sp, m, d, 0.5)
  sp.on_action(user_messages.ActionEvent(action='wheelie'))
  rlqw = run_steps(sp, m, d, 6.8)
  assert rlqw['upright'] and rlqw['max_front_lift_cm'] > 18.0, f'LQR wheelie failed: {rlqw}'
  sp.on_action(user_messages.ActionEvent(action='stoppie'))
  rlqs = run_steps(sp, m, d, 7.5)
  assert rlqs['upright'] and rlqs['max_rear_lift_cm'] > 16.0, f'LQR stoppie failed: {rlqs}'
  print('9a. LQR Baseline Cruise/Wheelie/Stoppie: PASS')

  sp, m, d = make_sim('slalom_cones', 'figure8', 3.4)
  res_c = run_steps(sp, m, d, 20.0)
  print('9. Slalom Cones Fig8:', res_c)
  assert res_c['upright'], 'Slalom cones fell!'

  # 10. Reset ride clears state and starts clean
  sp.on_action(user_messages.ActionEvent(action='reset_ride'))
  assert sp.distance_m == 0.0 and sp.active_trick == '' and sp.pending_trick == ''
  print('ALL VERIFICATION TESTS PASSED!')


if __name__ == '__main__':
  main()
