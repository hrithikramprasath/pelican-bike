// Client-side MuJoCo WASM physics controller (ported from sim_plugin.py).

import {
  FIG8_START_YAW,
  fig8Features,
  circleFeatures,
  cruiseFeatures,
  safeCornerSpeed,
} from './track.js';
import {
  WHEEL_RADIUS,
  CHAIN_RATIO,
  buildObsSingle,
  lqrSupervisorSingle,
} from './policy.js';

export const BIKE_PREFIX = 'pb_';
export const POLICY_SUBSTEPS = 10; // 50 Hz control (10 * 0.002s = 0.02s)

export const TRICK_LABELS = {
  turbo_boost: 'TURBO BOOST',
  wheelie: 'WHEELIE',
  wing_hop: 'WING-FLAP HOP',
  stoppie: 'STOPPIE',
  no_hands: 'NO-HANDS',
  s_carve: 'S-CARVE',
};

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function quatFromYaw(yaw) {
  return [Math.cos(0.5 * yaw), 0.0, 0.0, Math.sin(0.5 * yaw)];
}

export class PelicanSimController {
  constructor(policy) {
    this.policy = policy;
    this.mujoco = null;
    this.model = null;
    this.data = null;
    this.ids = null;

    // Live config
    this.mode = 'figure8';
    this.targetSpeed = 3.4;
    this.manualSteer = 0.0;
    this.wingMode = 'handlebars';
    this.flapIntensity = 0.75;
    this.beakOpenHold = false;
    this.autoHeadLook = true;
    this.useLearnedPolicy = true;

    // Path tracking state
    this.trackSIdx = null;
    this.cruiseAnchorX = 0.0;
    this.cruiseAnchorY = 0.0;
    this.cruiseAnchorYaw = 0.0;

    // Tricks & animated actions
    this.activeTrick = '';
    this.pendingTrick = '';
    this.trickStart = -1.0;
    this.trickPopAt = -1.0;
    this.trickUntil = -1.0;
    this.stoppieReleased = false;
    this.wheelieLanded = false;
    this.trickCount = 0;
    this.flapBurstUntil = -1.0;
    this.beakSnapUntil = -1.0;
    this.bellNodUntil = -1.0;

    // Telemetry state
    this.distanceM = 0.0;
    this.prevXyValid = false;
    this.prevX = 0.0;
    this.prevY = 0.0;
    this.roadErrM = 0.0;
    this.stepCounter = 0;

    // Held 50 Hz commands
    this.lastSteerCmd = 0.0;
    this.lastDriveCmd = 0.0;
    this.lastBrakeCmd = 0.0;
    this.lastPrCmd = 0.0;
    this.lastPpCmd = 0.04;
  }

  bindModel(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;

    const O = mujoco.mjtObj;
    const p = BIKE_PREFIX;
    const bodyId = (n) => mujoco.mj_name2id(model, O.mjOBJ_BODY.value, `${p}${n}`);
    const jntAdr = (n) => {
      const id = mujoco.mj_name2id(model, O.mjOBJ_JOINT.value, `${p}${n}`);
      return {
        id,
        q: model.jnt_qposadr[id],
        d: model.jnt_dofadr[id],
      };
    };
    const actId = (n) => mujoco.mj_name2id(model, O.mjOBJ_ACTUATOR.value, `${p}${n}`);
    const senAdr = (n) => model.sensor_adr[mujoco.mj_name2id(model, O.mjOBJ_SENSOR.value, `${p}${n}`)];

    this.ids = {
      bodies: {
        bike: bodyId('pelican_bicycle'),
        frontWheel: bodyId('front_wheel'),
        rearWheel: bodyId('rear_wheel'),
        wul: bodyId('wing_upper_left'),
        wur: bodyId('wing_upper_right'),
        wol: bodyId('wing_outer_left'),
        wor: bodyId('wing_outer_right'),
      },
      joints: {
        free: jntAdr('free'),
        rearWheel: jntAdr('rear_wheel_joint'),
        frontWheel: jntAdr('front_wheel_joint'),
        crank: jntAdr('crank_joint'),
        steer: jntAdr('steer_joint'),
        pelvisRoll: jntAdr('pelvis_roll'),
        pelvisPitch: jntAdr('pelvis_pitch'),
        hipLPitch: jntAdr('hip_l_pitch'),
        kneeLPitch: jntAdr('knee_l_pitch'),
        hipRPitch: jntAdr('hip_r_pitch'),
        kneeRPitch: jntAdr('knee_r_pitch'),
        beakOpen: jntAdr('beak_open'),
        wingLFlap: jntAdr('wing_l_flap'),
        wingRFlap: jntAdr('wing_r_flap'),
        wingLYaw: jntAdr('wing_l_yaw'),
      },
      act: {
        driveMotor: actId('drive_motor'),
        frontBrake: actId('front_brake'),
        steerServo: actId('steer_servo'),
        pelvisRollServo: actId('pelvis_roll_servo'),
        pelvisPitchServo: actId('pelvis_pitch_servo'),
        pedalLServo: actId('pedal_l_servo'),
        pedalRServo: actId('pedal_r_servo'),
        ankleLServo: actId('ankle_l_servo'),
        ankleRServo: actId('ankle_r_servo'),
        neckYawServo: actId('neck_yaw_servo'),
        headYawServo: actId('head_yaw_servo'),
        neckPitchServo: actId('neck_pitch_servo'),
        headPitchServo: actId('head_pitch_servo'),
        beakOpenServo: actId('beak_open_servo'),
        wingLYawServo: actId('wing_l_yaw_servo'),
        wingLFlapServo: actId('wing_l_flap_servo'),
        wingLPitchServo: actId('wing_l_pitch_servo'),
        wingLElbowServo: actId('wing_l_elbow_servo'),
        wingRYawServo: actId('wing_r_yaw_servo'),
        wingRFlapServo: actId('wing_r_flap_servo'),
        wingRPitchServo: actId('wing_r_pitch_servo'),
        wingRElbowServo: actId('wing_r_elbow_servo'),
      },
      sensors: {
        bikeVel: senAdr('bike_vel'),
        bikeGyro: senAdr('bike_gyro'),
      },
    };
  }

  initializeRollingState() {
    const { mujoco, model, data, ids } = this;
    mujoco.mj_resetData(model, data);

    const startYaw = this.mode === 'figure8' ? FIG8_START_YAW : 0.0;
    const v0 = Math.max(1.8, this.targetSpeed);
    const vx = v0 * Math.cos(startYaw);
    const vy = v0 * Math.sin(startYaw);

    const q = data.qpos;
    const v = data.qvel;
    const fq = ids.joints.free.q;
    const fd = ids.joints.free.d;
    const quat = quatFromYaw(startYaw);
    q[fq + 0] = 0.0;
    q[fq + 1] = 0.0;
    q[fq + 2] = 0.0;
    q[fq + 3] = quat[0];
    q[fq + 4] = quat[1];
    q[fq + 5] = quat[2];
    q[fq + 6] = quat[3];
    v[fd + 0] = vx;
    v[fd + 1] = vy;
    v[fd + 2] = 0.0;

    const omegaW = v0 / WHEEL_RADIUS;
    v[ids.joints.rearWheel.d] = omegaW;
    v[ids.joints.frontWheel.d] = omegaW;
    v[ids.joints.crank.d] = omegaW / CHAIN_RATIO;
    data.xfrc_applied.fill(0.0);

    mujoco.mj_forward(model, data);

    this.trackSIdx = this.mode === 'figure8' ? 0 : null;
    this.cruiseAnchorX = 0.0;
    this.cruiseAnchorY = 0.0;
    this.cruiseAnchorYaw = startYaw;
    this.activeTrick = '';
    this.pendingTrick = '';
    this.trickStart = -1.0;
    this.trickPopAt = -1.0;
    this.trickUntil = -1.0;
    this.stoppieReleased = false;
    this.wheelieLanded = false;
    this.flapBurstUntil = -1.0;
    this.beakSnapUntil = -1.0;
    this.bellNodUntil = -1.0;
    this.distanceM = 0.0;
    this.roadErrM = 0.0;
    this.stepCounter = 0;
    this.lastSteerCmd = 0.0;
    this.lastDriveCmd = 0.0;
    this.lastBrakeCmd = 0.0;
    this.lastPrCmd = 0.0;
    this.lastPpCmd = 0.04;
    this.prevXyValid = true;
    const bIdx = ids.bodies.bike * 3;
    this.prevX = data.xpos[bIdx + 0];
    this.prevY = data.xpos[bIdx + 1];
  }

  setMode(newMode) {
    if (newMode === this.mode) return;
    this.mode = newMode;
    this.trackSIdx = null;
    if (newMode === 'cruise' && this.data && this.ids) {
      const b3 = this.ids.bodies.bike * 3;
      const b9 = this.ids.bodies.bike * 9;
      const fx = this.data.xmat[b9 + 0];
      const fy = this.data.xmat[b9 + 3];
      this.cruiseAnchorX = this.data.xpos[b3 + 0];
      this.cruiseAnchorY = this.data.xpos[b3 + 1];
      this.cruiseAnchorYaw = Math.atan2(fy, fx);
    }
  }

  _triggerTrickNow(name, tNow) {
    const durations = {
      turbo_boost: 5.0,
      wheelie: 8.5,
      wing_hop: 7.5,
      stoppie: 8.5,
      no_hands: 3.8,
      s_carve: 4.2,
    };
    const snapDurs = {
      turbo_boost: 1.0,
      wheelie: 0.8,
      wing_hop: 0.8,
      stoppie: 0.7,
      no_hands: 1.2,
      s_carve: 0.0,
    };
    this.activeTrick = name;
    this.pendingTrick = '';
    this.trickStart = tNow;
    this.trickPopAt = name === 'wheelie' || name === 'stoppie' || name === 'wing_hop' ? -1.0 : tNow;
    this.trickUntil = tNow + durations[name];
    this.stoppieReleased = false;
    this.wheelieLanded = false;
    const sd = snapDurs[name] || 0.0;
    if (sd > 0.0) {
      this.beakSnapUntil = Math.max(this.beakSnapUntil, tNow + sd);
    }
  }

  triggerAction(act) {
    const { data, ids } = this;
    if (!data || !ids) return;
    const tNow = data.time;
    if (act in TRICK_LABELS) {
      const frontLift = data.xpos[ids.bodies.frontWheel * 3 + 2] - WHEEL_RADIUS;
      const rearLift = data.xpos[ids.bodies.rearWheel * 3 + 2] - WHEEL_RADIUS;
      const pitch = Math.asin(clamp(-data.xmat[ids.bodies.bike * 9 + 6], -1.0, 1.0));
      const isSettled = frontLift < 0.030 && rearLift < 0.030 && Math.abs(pitch) < 0.08;
      const midPopBusy =
        (this.activeTrick === 'wheelie' ||
          this.activeTrick === 'stoppie' ||
          this.activeTrick === 'wing_hop') &&
        this.trickPopAt >= 0.0 &&
        !this.wheelieLanded &&
        !this.stoppieReleased;
      if (!isSettled || midPopBusy) {
        this.pendingTrick = act;
      } else {
        this._triggerTrickNow(act, tNow);
      }
    } else if (act === 'flap_burst') {
      this.flapBurstUntil = tNow + 2.6;
      this.beakSnapUntil = Math.max(this.beakSnapUntil, tNow + 0.9);
    } else if (act === 'snap_beak') {
      this.beakSnapUntil = tNow + 0.85;
    } else if (act === 'ring_bell') {
      this.bellNodUntil = tNow + 0.65;
      this.beakSnapUntil = Math.max(this.beakSnapUntil, tNow + 0.50);
    } else if (act === 'nudge_left' || act === 'nudge_right') {
      const b9 = ids.bodies.bike * 9;
      const lx = data.xmat[b9 + 1];
      const ly = data.xmat[b9 + 4];
      const sign = act === 'nudge_left' ? 1.0 : -1.0;
      const fd = ids.joints.free.d;
      data.qvel[fd + 0] += sign * 0.55 * lx;
      data.qvel[fd + 1] += sign * 0.55 * ly;
      data.qvel[fd + 3] -= sign * 0.65;
    } else if (act === 'reset_ride') {
      this.initializeRollingState();
    }
  }

  _computeRoadFeatures(px, py, yaw, tNow) {
    if (this.mode === 'figure8') {
      const res = fig8Features(px, py, yaw, this.trackSIdx);
      this.trackSIdx = res.idx;
      return res;
    }
    if (this.mode === 'circle_left') {
      const r = circleFeatures(px, py, yaw, 1.0);
      return { ...r, k4: r.k3 };
    }
    if (this.mode === 'circle_right') {
      const r = circleFeatures(px, py, yaw, -1.0);
      return { ...r, k4: r.k3 };
    }
    if (this.mode === 'manual') {
      let kMan = 0.48 * clamp(this.manualSteer, -1.0, 1.0);
      if (this.activeTrick === 's_carve') {
        const elapsed = Math.max(0.0, tNow - this.trickStart);
        const env = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * Math.min(1.0, elapsed / 4.2)));
        kMan = clamp(kMan + 0.34 * env * Math.sin(2.0 * Math.PI * 0.55 * elapsed), -0.55, 0.55);
      } else if (
        (this.activeTrick === 'wheelie' ||
          this.activeTrick === 'stoppie' ||
          this.activeTrick === 'wing_hop') &&
        this.trickPopAt < 0.0
      ) {
        kMan *= 0.15;
      }
      return { eLat: 0.0, ePsi: 0.0, k0: kMan, k1: kMan, k2: kMan, k3: kMan, k4: kMan };
    }
    const r = cruiseFeatures(px, py, yaw, this.cruiseAnchorX, this.cruiseAnchorY, this.cruiseAnchorYaw);
    return { ...r, k4: r.k3 };
  }

  preStep() {
    const { data, ids, policy } = this;
    const t = data.time;
    data.xfrc_applied.fill(0.0);

    const b3 = ids.bodies.bike * 3;
    const b9 = ids.bodies.bike * 9;
    const px = data.xpos[b3 + 0];
    const py = data.xpos[b3 + 1];
    const m00 = data.xmat[b9 + 0], m02 = data.xmat[b9 + 2];
    const m10 = data.xmat[b9 + 3], m12 = data.xmat[b9 + 5];
    const m20 = data.xmat[b9 + 6], m22 = data.xmat[b9 + 8];

    const yaw = Math.atan2(m10, m00);
    const lhx = -Math.sin(yaw);
    const lhy = Math.cos(yaw);
    const lean = Math.atan2(-(m02 * lhx + m12 * lhy), Math.max(1e-4, m22));
    const pitch = Math.asin(clamp(-m20, -1.0, 1.0));

    const vAdr = ids.sensors.bikeVel;
    const gAdr = ids.sensors.bikeGyro;
    const vFwd = data.sensordata[vAdr + 0];
    const vLat = data.sensordata[vAdr + 1];
    const leanRate = data.sensordata[gAdr + 0];
    const pitchRate = data.sensordata[gAdr + 1];
    const yawRate = data.sensordata[gAdr + 2];

    const upright = Math.abs(lean) < 1.00 && m22 > 0.45;
    const frontLift = data.xpos[ids.bodies.frontWheel * 3 + 2] - WHEEL_RADIUS;
    const rearLift = data.xpos[ids.bodies.rearWheel * 3 + 2] - WHEEL_RADIUS;
    const groundedSettled = frontLift < 0.028 && rearLift < 0.028 && Math.abs(pitch) < 0.085;

    if (!upright) {
      this.activeTrick = '';
      this.pendingTrick = '';
      data.ctrl[ids.act.driveMotor] = 0.0;
      data.ctrl[ids.act.frontBrake] = 0.0;
      return;
    }

    if (this.activeTrick) {
      const earlyDone =
        this.pendingTrick !== '' &&
        this.trickPopAt >= 0.0 &&
        ((this.activeTrick === 'wheelie' && this.wheelieLanded) ||
          (this.activeTrick === 'stoppie' && this.stoppieReleased) ||
          (this.activeTrick === 'wing_hop' && t - this.trickPopAt > 0.65)) &&
        groundedSettled;
      const isPopTrick =
        this.activeTrick === 'wheelie' ||
        this.activeTrick === 'stoppie' ||
        this.activeTrick === 'wing_hop';
      const timeDone =
        t >= this.trickUntil && (!isPopTrick || groundedSettled || t >= this.trickUntil + 0.6);
      if (earlyDone || timeDone) {
        this.trickCount += 1;
        this.activeTrick = '';
      }
    }

    if (!this.activeTrick && this.pendingTrick && groundedSettled) {
      this._triggerTrickNow(this.pendingTrick, t);
    }

    const rf = this._computeRoadFeatures(px, py, yaw, t);
    const eLatRaw = rf.eLat;
    const ePsi = rf.ePsi;
    let { k0, k1, k2, k3, k4 } = rf;
    this.roadErrM = Math.abs(eLatRaw);

    const trick = this.activeTrick;
    const trickElapsed = trick ? Math.max(0.0, t - this.trickStart) : 0.0;

    let eLat = eLatRaw;
    let vReq = this.targetSpeed;
    if (trick === 'turbo_boost') {
      vReq = Math.max(vReq, 5.8);
    } else if (trick === 's_carve') {
      const env = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * Math.min(1.0, trickElapsed / 4.2)));
      eLat = eLatRaw - 0.55 * env * Math.sin(2.0 * Math.PI * 0.55 * trickElapsed);
    }

    let vTgt;
    if (trick === 'turbo_boost') {
      const maxKb = Math.max(
        Math.abs(k0),
        Math.abs(k1),
        Math.abs(k2),
        0.80 * Math.abs(k3),
        0.55 * Math.abs(k4)
      );
      vTgt = maxKb > 1e-4 ? Math.max(2.1, Math.min(vReq, Math.sqrt(5.0 / maxKb))) : vReq;
    } else {
      vTgt = safeCornerSpeed(vReq, k0, k1, k2, k3, k4);
      if (this.mode === 'manual' && Math.abs(k0) > 0.22) {
        vTgt = Math.max(vTgt, 2.20);
      }
    }

    if ((trick === 'wheelie' || trick === 'stoppie' || trick === 'wing_hop') && this.trickPopAt < 0.0) {
      if (trick === 'wheelie') {
        if (Math.abs(k1) < 0.32 && Math.abs(k2) < 0.34) {
          vTgt = Math.min(vTgt, 2.15);
        }
      } else if (trick === 'stoppie') {
        const maxKNear = Math.max(Math.abs(k0), Math.abs(k1), Math.abs(k2));
        vTgt =
          maxKNear < 0.28
            ? Math.max(vTgt, 4.75)
            : safeCornerSpeed(Math.max(vReq, 4.75), k0, k1, k2, k3, k4);
      }
      if (this.mode === 'circle_left' || this.mode === 'circle_right') {
        k0 = 0.0;
        k1 = 0.0;
        k2 = 0.0;
        k3 = 0.0;
        k4 = 0.0;
      }

      const leanLim = trick === 'wheelie' || trick === 'stoppie' ? 0.135 : 0.14;
      const rateLim = trick === 'wheelie' ? 0.28 : trick === 'stoppie' ? 1.15 : 0.65;
      const latLim = trick === 'wheelie' ? 0.68 : 0.58;
      const minStopV = this.mode !== 'manual' ? 3.95 : 3.55;
      let speedOk = true;
      if (trick === 'stoppie') speedOk = vFwd > minStopV;
      else if (trick === 'wheelie' && this.mode === 'figure8') speedOk = vFwd < 2.48;

      const k0Lim = 0.20;
      const k1Lim = trick === 'wheelie' ? 0.23 : 0.24;
      const k2Lim = trick === 'wheelie' ? 0.24 : 0.26;
      const k3Lim = trick === 'wheelie' ? 0.34 : 0.38;

      const isStraight =
        this.mode === 'figure8' &&
        speedOk &&
        Math.abs(lean) < leanLim &&
        Math.abs(leanRate) < rateLim &&
        Math.abs(k0) < k0Lim &&
        Math.abs(k1) < k1Lim &&
        Math.abs(k2) < k2Lim &&
        Math.abs(k3) < k3Lim &&
        Math.abs(eLatRaw) < latLim;

      const nonFig8Lean = trick === 'wing_hop' ? 0.22 : 0.12;
      const nonFig8Rate = trick === 'wing_hop' ? 0.35 : 0.24;
      const isNonFig8Ok =
        this.mode !== 'figure8' &&
        Math.abs(lean) < nonFig8Lean &&
        Math.abs(leanRate) < nonFig8Rate &&
        speedOk;

      const durPop = trick === 'wing_hop' ? 1.35 : 1.05;
      if ((isStraight || isNonFig8Ok) && groundedSettled) {
        this.trickPopAt = t;
        this.trickUntil = t + durPop;
      } else if (trickElapsed > 7.2) {
        if (
          Math.abs(lean) < 0.28 &&
          Math.abs(k1) < 0.36 &&
          (vFwd > 3.15 || trick !== 'stoppie') &&
          groundedSettled
        ) {
          this.trickPopAt = t;
          this.trickUntil = t + durPop;
        } else {
          this.activeTrick = '';
        }
      }
    }

    const popElapsed = this.trickPopAt >= 0.0 ? t - this.trickPopAt : -1.0;

    let obsTargetPitch = 0.0;
    let obsWheelieActive = 0.0;
    if (trick === 'wheelie' && popElapsed >= 0.0 && !this.wheelieLanded) {
      const popRem = Math.max(0.0, this.trickUntil - t);
      const wRamp = Math.max(0.0, Math.min(1.0, popElapsed / 0.18, popRem / 0.38));
      obsTargetPitch = -0.225 * wRamp;
      obsWheelieActive = 1.0;
    }
    const obsNoHands = trick === 'no_hands' ? 1.0 : 0.0;

    if (this.stepCounter % POLICY_SUBSTEPS === 0) {
      const obs = buildObsSingle({
        lean,
        leanRate,
        pitch,
        pitchRate,
        yawRate,
        vFwd,
        vLat,
        vTarget: vTgt,
        steerQ: data.qpos[ids.joints.steer.q],
        steerQd: data.qvel[ids.joints.steer.d],
        pelvisRollQ: data.qpos[ids.joints.pelvisRoll.q],
        pelvisPitchQ: data.qpos[ids.joints.pelvisPitch.q],
        eLat,
        ePsi,
        k0,
        k1,
        k2,
        k3,
        targetPitch: obsTargetPitch,
        wheelieActive: obsWheelieActive,
        noHandsActive: obsNoHands,
      });
      const crankQd = data.qvel[ids.joints.crank.d];

      let act;
      if (this.useLearnedPolicy && policy) {
        act = policy.act(obs);
        const oodW = Math.max(
          clamp((Math.abs(eLatRaw) - 5.2) / 1.0, 0.0, 1.0),
          clamp((Math.abs(lean) - 0.52) / 0.08, 0.0, 1.0)
        );
        if (oodW > 0.0) {
          const actLqr = lqrSupervisorSingle(obs, crankQd, policy.lqrSpeeds, policy.lqrK);
          act = act.map((v, i) => (1.0 - oodW) * v + oodW * actLqr[i]);
        }
      } else {
        act = lqrSupervisorSingle(obs, crankQd, policy.lqrSpeeds, policy.lqrK);
      }

      let steerCmd = act[0];
      let driveCmd = Math.max(-65.0, act[1]);
      if (trick !== 'stoppie' && vFwd < 1.85) {
        driveCmd = Math.max(driveCmd, Math.min(115.0, 50.0 * (2.05 - vFwd)));
      }
      let prCmd = act[2];
      let ppCmd = act[3];
      const brakeMargin = trick === 'wheelie' && this.trickPopAt < 0.0 ? 0.20 : 0.55;
      let brakeCmd =
        vFwd > vTgt + brakeMargin &&
        trick !== 'turbo_boost' &&
        obsWheelieActive < 0.5 &&
        frontLift < 0.02 &&
        rearLift < 0.02 &&
        Math.abs(lean) < 0.42
          ? clamp(28.0 * (vTgt - vFwd), -65.0, 0.0)
          : 0.0;

      if (trick === 'turbo_boost') {
        ppCmd = 0.16;
        if (vFwd < vTgt) {
          driveCmd = Math.max(driveCmd, Math.min(155.0, 28.0 * (vTgt - vFwd) + 8.0));
        }
      } else if (trick === 'wheelie' && popElapsed >= 0.0) {
        const popRem = Math.max(0.0, this.trickUntil - t);
        const ramp = Math.max(0.0, Math.min(1.0, popElapsed / 0.08, popRem / 0.38));
        const leanSafe = Math.max(0.0, 1.0 - Math.max(0.0, Math.abs(lean) - 0.16) / 0.16);
        const curveSafe = Math.abs(k0) < 0.28 && (this.mode === 'figure8' ? Math.abs(k1) < 0.36 : true);
        const rollGrowing = Math.abs(lean) > 0.16 || (Math.abs(lean) > 0.08 && lean * leanRate > 0.025);
        if (!this.wheelieLanded) {
          if (
            popElapsed < 0.50 &&
            vFwd < 4.55 &&
            leanSafe > 0.20 &&
            !rollGrowing &&
            curveSafe &&
            pitch > -0.45
          ) {
            const thetaDes = -0.220 * ramp * leanSafe;
            const popBoost = 132.0 * clamp((pitch + 0.14) / 0.10, 0.0, 1.0);
            const wDrive = clamp(
              (104.0 + popBoost) * leanSafe + 430.0 * (pitch - thetaDes) + 110.0 * pitchRate,
              -45.0,
              270.0
            );
            if (this.useLearnedPolicy) {
              const blendDrive = 0.20 * act[1] + 0.80 * wDrive;
              driveCmd = pitch < -0.06 ? Math.min(blendDrive, wDrive + 5.0) : blendDrive;
            } else {
              driveCmd = wDrive;
            }
            driveCmd = clamp(driveCmd, -45.0, 270.0);
            brakeCmd = 0.0;
          } else {
            this.wheelieLanded = true;
          }
        }
        if (!this.wheelieLanded) {
          ppCmd = -0.35 * ramp;
          prCmd = clamp(1.8 * lean + 0.32 * leanRate, -0.35, 0.35);
          if (frontLift > 0.035) {
            steerCmd = clamp(-0.95 * lean - 0.16 * leanRate, -0.42, 0.42);
          } else if (popElapsed < 0.22) {
            steerCmd = clamp(-0.75 * lean - 0.14 * leanRate, -0.28, 0.28);
          }
        } else {
          ppCmd = 0.04;
          if (frontLift > 0.06) {
            driveCmd = Math.max(-20.0, Math.min(driveCmd, 10.0));
            steerCmd = clamp(-0.95 * lean - 0.16 * leanRate, -0.42, 0.42);
          }
        }
      } else if (trick === 'stoppie' && popElapsed >= 0.0) {
        if (
          !this.stoppieReleased &&
          popElapsed < 0.55 &&
          vFwd > 1.32 &&
          pitch < 0.34 &&
          Math.abs(lean) < 0.22
        ) {
          const sRamp = Math.max(0.0, Math.min(1.0, popElapsed / 0.06, (0.55 - popElapsed) / 0.14));
          const tgtP = 0.25 * sRamp;
          driveCmd = 0.0;
          brakeCmd = clamp(-95.0 * sRamp - 460.0 * (tgtP - pitch) + 60.0 * pitchRate, -205.0, 0.0);
          ppCmd = 0.34 * sRamp;
          prCmd = clamp(1.8 * lean + 0.32 * leanRate, -0.35, 0.35);
          steerCmd = clamp(0.45 * steerCmd - 0.65 * lean - 0.12 * leanRate, -0.26, 0.26);
        } else {
          this.stoppieReleased = true;
          brakeCmd = 0.0;
          driveCmd = Math.max(driveCmd, Math.min(155.0, 48.0 * Math.max(0.0, vTgt - vFwd)));
          ppCmd = 0.04;
        }
      } else if (trick === 'wing_hop' && popElapsed >= 0.0) {
        ppCmd = popElapsed < 0.18 ? 0.16 : popElapsed < 0.46 ? -0.06 : 0.04;
        if (frontLift > 0.02 && rearLift > 0.02) {
          steerCmd = clamp(0.45 * k0 - 0.35 * ePsi - 0.55 * lean - 0.10 * leanRate, -0.24, 0.24);
          prCmd = clamp(1.3 * lean + 0.22 * leanRate, -0.32, 0.32);
        }
      } else if (trick === 'no_hands') {
        ppCmd = -0.14;
      }

      if (frontLift > 0.035 && rearLift > 0.035) {
        steerCmd = clamp(-0.50 * lean - 0.08 * leanRate, -0.22, 0.22);
        prCmd = clamp(1.1 * lean + 0.20 * leanRate, -0.30, 0.30);
      }

      this.lastSteerCmd = steerCmd;
      this.lastDriveCmd = driveCmd;
      this.lastBrakeCmd = brakeCmd;
      this.lastPrCmd = prCmd;
      this.lastPpCmd = ppCmd;
    }

    this.stepCounter += 1;

    if (trick === 'wing_hop' && popElapsed >= 0.18 && popElapsed < 0.42) {
      const qdL = data.qvel[ids.joints.wingLFlap.d];
      const qdR = data.qvel[ids.joints.wingRFlap.d];
      const liftBase = Math.min(152.0, 116.0 + 15.0 * Math.max(0.0, -0.5 * (qdL + qdR)));
      const rollCorr = clamp(1.15 * lean + 0.22 * leanRate, -0.22, 0.22);
      const liftL = liftBase * (1.0 - rollCorr);
      const liftR = liftBase * (1.0 + rollCorr);
      data.xfrc_applied[ids.bodies.wul * 6 + 2] = 0.55 * liftL;
      data.xfrc_applied[ids.bodies.wur * 6 + 2] = 0.55 * liftR;
      data.xfrc_applied[ids.bodies.wol * 6 + 2] = 0.45 * liftL;
      data.xfrc_applied[ids.bodies.wor * 6 + 2] = 0.45 * liftR;
    }

    const A = ids.act;
    data.ctrl[A.steerServo] = this.lastSteerCmd;
    data.ctrl[A.driveMotor] = this.lastDriveCmd;
    data.ctrl[A.frontBrake] = this.lastBrakeCmd;
    data.ctrl[A.pelvisRollServo] = this.lastPrCmd;
    data.ctrl[A.pelvisPitchServo] = this.lastPpCmd;

    const crankQ = data.qpos[ids.joints.crank.q];
    const crankQd = data.qvel[ids.joints.crank.d];
    const cqMid = crankQ + 0.002 * crankQd;
    data.ctrl[A.pedalLServo] = -cqMid;
    data.ctrl[A.pedalRServo] = -cqMid;

    const pelvisP = data.qpos[ids.joints.pelvisPitch.q];
    data.ctrl[A.ankleLServo] = -(
      data.qpos[ids.joints.hipLPitch.q] +
      data.qpos[ids.joints.kneeLPitch.q] +
      pelvisP
    );
    data.ctrl[A.ankleRServo] = -(
      data.qpos[ids.joints.hipRPitch.q] +
      data.qpos[ids.joints.kneeRPitch.q] +
      pelvisP
    );

    if (this.autoHeadLook) {
      data.ctrl[A.neckYawServo] = 0.36 * this.lastSteerCmd;
      data.ctrl[A.headYawServo] = 0.48 * this.lastSteerCmd;
    } else {
      data.ctrl[A.neckYawServo] = 0.0;
      data.ctrl[A.headYawServo] = 0.0;
    }

    let neckPitchCmd = -0.05 * Math.sin(2.0 * crankQ);
    let headNod = 0.04 * Math.sin(2.0 * crankQ);
    if (trick === 'turbo_boost') {
      neckPitchCmd = 0.26;
      headNod = -0.22;
    } else if (trick === 'wheelie') {
      neckPitchCmd = -0.22;
      headNod = 0.14;
    } else if (trick === 'stoppie' && popElapsed >= 0.0 && !this.stoppieReleased) {
      neckPitchCmd = -0.38;
      headNod = -0.18;
    }
    if (t < this.bellNodUntil) {
      headNod += 0.22 * Math.sin(18.0 * (this.bellNodUntil - t));
    }
    data.ctrl[A.neckPitchServo] = neckPitchCmd;
    data.ctrl[A.headPitchServo] = headNod;

    let beakTarget;
    if (t < this.beakSnapUntil) {
      const rem = this.beakSnapUntil - t;
      beakTarget = 0.42 * Math.max(0.0, Math.sin(14.0 * rem)) + 0.14;
    } else if (this.beakOpenHold || trick === 'turbo_boost' || trick === 'no_hands') {
      beakTarget = 0.44 + 0.05 * Math.sin(4.0 * t);
    } else {
      beakTarget = 0.06 + 0.05 * Math.sin(2.2 * t);
    }
    data.ctrl[A.beakOpenServo] = beakTarget;

    if (trick === 'wing_hop' && popElapsed >= 0.0) {
      const wFlap = popElapsed < 0.18 ? 1.10 : popElapsed < 0.46 ? -0.70 : 0.25;
      const wYaw = popElapsed < 0.46 ? 0.35 : 1.08;
      const wEl = popElapsed < 0.18 ? -0.20 : popElapsed < 0.46 ? 0.35 : 0.05;
      data.ctrl[A.wingLYawServo] = wYaw;
      data.ctrl[A.wingRYawServo] = wYaw;
      data.ctrl[A.wingLFlapServo] = wFlap;
      data.ctrl[A.wingRFlapServo] = wFlap;
      data.ctrl[A.wingLPitchServo] = -0.10;
      data.ctrl[A.wingRPitchServo] = -0.10;
      data.ctrl[A.wingLElbowServo] = wEl;
      data.ctrl[A.wingRElbowServo] = wEl;
    } else if (trick === 'turbo_boost') {
      if (trickElapsed < 0.55) {
        const phase = 11.0 * trickElapsed;
        data.ctrl[A.wingLYawServo] = 1.15;
        data.ctrl[A.wingRYawServo] = 1.15;
        data.ctrl[A.wingLFlapServo] = 0.20 + 0.70 * Math.sin(phase);
        data.ctrl[A.wingRFlapServo] = 0.20 + 0.70 * Math.sin(phase);
        data.ctrl[A.wingLPitchServo] = -0.18;
        data.ctrl[A.wingRPitchServo] = -0.18;
        data.ctrl[A.wingLElbowServo] = -0.45 * Math.cos(phase);
        data.ctrl[A.wingRElbowServo] = -0.45 * Math.cos(phase);
      } else {
        data.ctrl[A.wingLYawServo] = 1.28;
        data.ctrl[A.wingRYawServo] = 1.28;
        data.ctrl[A.wingLFlapServo] = 0.06;
        data.ctrl[A.wingRFlapServo] = 0.06;
        data.ctrl[A.wingLPitchServo] = -0.22;
        data.ctrl[A.wingRPitchServo] = -0.22;
        data.ctrl[A.wingLElbowServo] = 0.10;
        data.ctrl[A.wingRElbowServo] = 0.10;
      }
    } else if (trick === 'wheelie' || trick === 'stoppie' || trick === 'no_hands') {
      const phase = 6.5 * trickElapsed;
      data.ctrl[A.wingLYawServo] = 1.10;
      data.ctrl[A.wingRYawServo] = 1.10;
      data.ctrl[A.wingLFlapServo] =
        trick === 'no_hands' ? 0.40 + 0.30 * Math.sin(phase) : 0.30 - 0.35 * lean;
      data.ctrl[A.wingRFlapServo] =
        trick === 'no_hands' ? 0.40 + 0.30 * Math.sin(phase) : 0.30 + 0.35 * lean;
      data.ctrl[A.wingLPitchServo] = -0.10;
      data.ctrl[A.wingRPitchServo] = -0.10;
      data.ctrl[A.wingLElbowServo] = 0.05;
      data.ctrl[A.wingRElbowServo] = 0.05;
    } else if (trick === 's_carve') {
      data.ctrl[A.wingLYawServo] = 1.05;
      data.ctrl[A.wingRYawServo] = 1.05;
      data.ctrl[A.wingLFlapServo] = 0.24 - 0.65 * lean;
      data.ctrl[A.wingRFlapServo] = 0.24 + 0.65 * lean;
      data.ctrl[A.wingLPitchServo] = -0.08;
      data.ctrl[A.wingRPitchServo] = -0.08;
      data.ctrl[A.wingLElbowServo] = 0.05;
      data.ctrl[A.wingRElbowServo] = 0.05;
    } else if (this.wingMode === 'flap' || t < this.flapBurstUntil) {
      const amp = this.wingMode === 'flap' ? this.flapIntensity : 0.92;
      const phase = 7.2 * t;
      const flap = 0.18 + 0.72 * amp * Math.sin(phase);
      const elbow = -0.52 * amp * Math.cos(phase);
      data.ctrl[A.wingLYawServo] = 1.05;
      data.ctrl[A.wingRYawServo] = 1.05;
      data.ctrl[A.wingLFlapServo] = flap;
      data.ctrl[A.wingRFlapServo] = flap;
      data.ctrl[A.wingLPitchServo] = -0.15 * amp * Math.sin(phase);
      data.ctrl[A.wingRPitchServo] = -0.15 * amp * Math.sin(phase);
      data.ctrl[A.wingLElbowServo] = elbow;
      data.ctrl[A.wingRElbowServo] = elbow;
    } else if (this.wingMode === 'glide') {
      data.ctrl[A.wingLYawServo] = 1.12;
      data.ctrl[A.wingRYawServo] = 1.12;
      data.ctrl[A.wingLFlapServo] = 0.22 - 0.65 * lean;
      data.ctrl[A.wingRFlapServo] = 0.22 + 0.65 * lean;
      data.ctrl[A.wingLPitchServo] = -0.08;
      data.ctrl[A.wingRPitchServo] = -0.08;
      data.ctrl[A.wingLElbowServo] = 0.05;
      data.ctrl[A.wingRElbowServo] = 0.05;
    } else {
      const steerQ = data.qpos[ids.joints.steer.q];
      data.ctrl[A.wingLYawServo] = -0.22 * steerQ;
      data.ctrl[A.wingRYawServo] = 0.22 * steerQ;
      data.ctrl[A.wingLFlapServo] = 0.04 * Math.sin(2.0 * crankQ);
      data.ctrl[A.wingRFlapServo] = -0.04 * Math.sin(2.0 * crankQ);
      data.ctrl[A.wingLPitchServo] = 0.16 * steerQ;
      data.ctrl[A.wingRPitchServo] = -0.16 * steerQ;
      data.ctrl[A.wingLElbowServo] = 0.0;
      data.ctrl[A.wingRElbowServo] = 0.0;
    }
  }

  postStep() {
    const { data, ids } = this;
    const b3 = ids.bodies.bike * 3;
    const px = data.xpos[b3 + 0];
    const py = data.xpos[b3 + 1];
    if (this.prevXyValid) {
      const stepDist = Math.hypot(px - this.prevX, py - this.prevY);
      if (stepDist < 0.5) {
        this.distanceM += stepDist;
      }
    }
    this.prevX = px;
    this.prevY = py;
    this.prevXyValid = true;
  }

  getTelemetry() {
    const { data, ids, policy } = this;
    const b3 = ids.bodies.bike * 3;
    const b9 = ids.bodies.bike * 9;
    const px = data.xpos[b3 + 0];
    const py = data.xpos[b3 + 1];
    const m00 = data.xmat[b9 + 0], m02 = data.xmat[b9 + 2];
    const m10 = data.xmat[b9 + 3], m12 = data.xmat[b9 + 5];
    const m20 = data.xmat[b9 + 6], m22 = data.xmat[b9 + 8];

    const yaw = Math.atan2(m10, m00);
    const lhx = -Math.sin(yaw);
    const lhy = Math.cos(yaw);
    const lean = Math.atan2(-(m02 * lhx + m12 * lhy), Math.max(1e-4, m22));
    const pitch = Math.asin(clamp(-m20, -1.0, 1.0));

    const vFwd = data.sensordata[ids.sensors.bikeVel + 0];
    const steer = data.qpos[ids.joints.steer.q];
    const omegaCrank = data.qvel[ids.joints.crank.d];
    const frontLift = data.xpos[ids.bodies.frontWheel * 3 + 2] - WHEEL_RADIUS;
    const rearLift = data.xpos[ids.bodies.rearWheel * 3 + 2] - WHEEL_RADIUS;

    const itN = Math.round(policy?.metadata?.iter ?? 0);
    const errN = (policy?.metadata?.eval_err_m ?? 0).toFixed(2);
    const radToDeg = 180.0 / Math.PI;

    return {
      simTime: data.time,
      speedMps: vFwd,
      leanDeg: lean * radToDeg,
      pitchDeg: -pitch * radToDeg,
      steerDeg: steer * radToDeg,
      crankRpm: (omegaCrank * 60.0) / (2.0 * Math.PI),
      distanceM: this.distanceM,
      roadErrM: this.roadErrM,
      posX: px,
      posY: py,
      upright: Math.abs(lean) < 1.00 && m22 > 0.45,
      airborne: frontLift > 0.04 && rearLift > 0.04,
      frontLiftCm: frontLift * 100.0,
      rearLiftCm: rearLift * 100.0,
      activeTrick: TRICK_LABELS[this.activeTrick] || '',
      pendingTrick: TRICK_LABELS[this.pendingTrick] || '',
      trickCount: this.trickCount,
      policyLabel: this.useLearnedPolicy
        ? `Trained PPO MLP (iter ${itN}, eval ${errN}m)`
        : 'LQR Baseline',
    };
  }
}
