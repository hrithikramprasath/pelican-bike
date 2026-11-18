// Pure-JS MLP policy & LQR safety supervisor for Pelican Bicycle (ported from policy.py).

import { wrapPi } from './track.js';

export const OBS_DIM = 22;
export const ACT_DIM = 4;

export const ACT_SCALE = [0.72, 300.0, 0.36, 0.37];
export const ACT_BIAS = [0.0, 0.0, 0.0, -0.01];

export const WHEEL_RADIUS = 0.330;
export const WHEELBASE = 1.090;
export const HEAD_ANGLE_RAD = (20.0 * Math.PI) / 180.0;
export const CHAIN_RATIO = 2.0;

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function interp1d(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  for (let i = 0; i < n - 1; i++) {
    if (x >= xs[i] && x <= xs[i + 1]) {
      const t = (x - xs[i]) / Math.max(1e-9, xs[i + 1] - xs[i]);
      return ys[i] + t * (ys[i + 1] - ys[i]);
    }
  }
  return ys[n - 1];
}

export function buildObsSingle({
  lean,
  leanRate,
  pitch,
  pitchRate,
  yawRate,
  vFwd,
  vLat,
  vTarget,
  steerQ,
  steerQd,
  pelvisRollQ,
  pelvisPitchQ,
  eLat,
  ePsi,
  k0,
  k1,
  k2,
  k3,
  targetPitch = 0.0,
  wheelieActive = 0.0,
  noHandsActive = 0.0,
}) {
  return new Float64Array([
    lean,
    leanRate,
    pitch,
    pitchRate,
    yawRate,
    vFwd,
    vLat,
    vTarget,
    vTarget - vFwd,
    steerQ,
    steerQd,
    pelvisRollQ,
    pelvisPitchQ,
    clamp(eLat, -6.0, 6.0),
    ePsi,
    k0,
    k1,
    k2,
    k3,
    targetPitch,
    wheelieActive,
    noHandsActive,
  ]);
}

export function lqrSupervisorSingle(obs, crankQvel, lqrSpeeds, lqrK) {
  const lean = obs[0];
  const leanRate = obs[1];
  const pitch = obs[2];
  const pitchRate = obs[3];
  const vFwd = obs[5];
  const vTgt = obs[7];
  const steerQ = obs[9];
  const steerQd = obs[10];
  const eLat = obs[13];
  const ePsi = obs[14];
  const k0 = obs[15];
  const k1 = obs[16];
  const k2 = obs[17];
  const targetPitch = obs[19];
  const wheelie = obs[20] > 0.5;

  const vEff = Math.max(1.6, Math.abs(vFwd));
  const kFf = 0.25 * k0 + 0.40 * k1 + 0.35 * k2;
  const lLook = Math.abs(eLat) < 1.2
    ? Math.max(2.8, 1.05 * vEff)
    : Math.max(2.2, 0.78 * vEff);
  const psiDes = Math.atan2(-eLat, lLook);
  const ePsiEff = wrapPi(ePsi - psiDes);
  const onTrack = Math.exp(-0.25 * eLat * eLat);
  const aLat = clamp(-1.7 * vEff * ePsiEff, -4.3, 4.3);
  const kDes = clamp(onTrack * kFf + aLat / (vEff * vEff), -0.68, 0.68);
  let phiDes = clamp(-Math.atan((vEff * vEff * kDes) / 9.81), -0.45, 0.45);
  if (wheelie) phiDes *= 0.30;

  const kActual = -(9.81 * Math.tan(clamp(lean, -0.58, 0.58))) / (vEff * vEff);
  const deltaActual = Math.atan(WHEELBASE * kActual) / Math.cos(HEAD_ANGLE_RAD);

  const kCol0 = lqrK.map((row) => row[0]);
  const kCol1 = lqrK.map((row) => row[1]);
  const kCol2 = lqrK.map((row) => row[2]);
  const kCol3 = lqrK.map((row) => row[3]);
  const kGains = [
    interp1d(vEff, lqrSpeeds, kCol0),
    interp1d(vEff, lqrSpeeds, kCol1),
    interp1d(vEff, lqrSpeeds, kCol2),
    interp1d(vEff, lqrSpeeds, kCol3),
  ];

  const xErr = [lean - phiDes, leanRate, steerQ - deltaActual, steerQd];
  const kDotX =
    kGains[0] * xErr[0] +
    kGains[1] * xErr[1] +
    kGains[2] * xErr[2] +
    kGains[3] * xErr[3];
  const steerLqr = clamp(deltaActual - kDotX, -0.70, 0.70);
  const frontAir = wheelie && pitch < -0.04;
  const steerAir = clamp(-0.55 * lean - 0.10 * leanRate, -0.25, 0.25);
  const steerCmd = frontAir ? steerAir : steerLqr;

  const omegaCDes = vTgt / WHEEL_RADIUS / CHAIN_RATIO;
  const driveCruise = clamp(
    22.0 * (vTgt - vFwd) + 4.0 * (omegaCDes - crankQvel),
    -140.0,
    140.0
  );
  const leanSafe = clamp(1.0 - Math.max(0.0, Math.abs(lean) - 0.18) / 0.18, 0.0, 1.0);
  const speedSafe = clamp((5.9 - vFwd) / 0.7, 0.0, 1.0);
  const effTargetPitch = targetPitch * leanSafe * speedSafe;
  const driveWheelie = clamp(
    145.0 * leanSafe * speedSafe +
      480.0 * (pitch - effTargetPitch) +
      65.0 * pitchRate,
    -40.0,
    300.0
  );
  const driveCmd = wheelie ? driveWheelie : driveCruise;

  const prCmd = wheelie ? clamp(1.8 * lean + 0.32 * leanRate, -0.35, 0.35) : 0.0;
  const ppCmd = wheelie ? clamp(effTargetPitch - 0.12, -0.36, 0.34) : 0.04;

  const raw = [steerCmd, driveCmd, prCmd, ppCmd];
  return raw.map((val, idx) => {
    const n = clamp((val - ACT_BIAS[idx]) / ACT_SCALE[idx], -0.995, 0.995);
    return n * ACT_SCALE[idx] + ACT_BIAS[idx];
  });
}

export class PelicanPolicy {
  constructor(json) {
    this.obsMean = Float64Array.from(json.obs_mean);
    this.obsStd = Float64Array.from(json.obs_std.map((s) => Math.max(1e-3, s)));
    // Transpose w matrices to row-major output-first [outDim][inDim] or keep [inDim][outDim]
    this.w1 = json.w1.map((row) => Float64Array.from(row)); // (22, 96)
    this.b1 = Float64Array.from(json.b1);
    this.w2 = json.w2.map((row) => Float64Array.from(row)); // (96, 96)
    this.b2 = Float64Array.from(json.b2);
    this.w3 = json.w3.map((row) => Float64Array.from(row)); // (96, 4)
    this.b3 = Float64Array.from(json.b3);
    this.lqrSpeeds = json.lqr_speeds;
    this.lqrK = json.lqr_k;
    this.metadata = {
      iter: json.iter ?? 0,
      eval_err_m: json.eval_err_m ?? 0,
      eval_surv_s: json.eval_surv_s ?? 0,
      eval_wheelie_deg: json.eval_wheelie_deg ?? 0,
    };

    // Pre-allocated scratch buffers so inference produces zero GC pressure at 50 Hz
    this._x = new Float64Array(OBS_DIM);
    this._h1 = new Float64Array(this.b1.length);
    this._h2 = new Float64Array(this.b2.length);
    this._out = new Float64Array(ACT_DIM);
  }

  act(obs) {
    const x = this._x;
    for (let i = 0; i < OBS_DIM; i++) {
      x[i] = clamp((obs[i] - this.obsMean[i]) / this.obsStd[i], -8.0, 8.0);
    }

    const h1 = this._h1;
    const h1Len = h1.length;
    h1.set(this.b1);
    for (let i = 0; i < OBS_DIM; i++) {
      const xi = x[i];
      const row = this.w1[i];
      for (let j = 0; j < h1Len; j++) {
        h1[j] += xi * row[j];
      }
    }
    for (let j = 0; j < h1Len; j++) {
      h1[j] = Math.tanh(h1[j]);
    }

    const h2 = this._h2;
    const h2Len = h2.length;
    h2.set(this.b2);
    for (let i = 0; i < h1Len; i++) {
      const hi = h1[i];
      const row = this.w2[i];
      for (let j = 0; j < h2Len; j++) {
        h2[j] += hi * row[j];
      }
    }
    for (let j = 0; j < h2Len; j++) {
      h2[j] = Math.tanh(h2[j]);
    }

    const out = this._out;
    out.set(this.b3);
    for (let i = 0; i < h2Len; i++) {
      const hi = h2[i];
      const row = this.w3[i];
      for (let j = 0; j < ACT_DIM; j++) {
        out[j] += hi * row[j];
      }
    }
    for (let j = 0; j < ACT_DIM; j++) {
      out[j] = Math.tanh(out[j]) * ACT_SCALE[j] + ACT_BIAS[j];
    }
    return [out[0], out[1], out[2], out[3]];
  }
}
