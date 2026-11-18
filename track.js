// Continuous phase-locked road geometry for Pelican Bike (ported from track.py).

export const FIG8_A = 7.5;
export const FIG8_B = 3.8;
export const FIG8_START_YAW = Math.atan2(2.0 * FIG8_B, FIG8_A);
export const CIRCLE_RADIUS = 4.6;

const N_FIG8 = 720;
const DS = (2.0 * Math.PI) / N_FIG8;

const FIG8_X = new Float64Array(N_FIG8);
const FIG8_Y = new Float64Array(N_FIG8);
const FIG8_YAWS = new Float64Array(N_FIG8);
const FIG8_TX = new Float64Array(N_FIG8);
const FIG8_TY = new Float64Array(N_FIG8);
const FIG8_KAPPAS = new Float64Array(N_FIG8);
const ARC_SEG = new Float64Array(N_FIG8);

for (let i = 0; i < N_FIG8; i++) {
  const s = i * DS;
  const x = FIG8_A * Math.sin(s);
  const y = FIG8_B * Math.sin(2.0 * s);
  const dx = FIG8_A * Math.cos(s);
  const dy = 2.0 * FIG8_B * Math.cos(2.0 * s);
  const ddx = -FIG8_A * Math.sin(s);
  const ddy = -4.0 * FIG8_B * Math.sin(2.0 * s);
  const speedDs = Math.hypot(dx, dy);
  const yaw = Math.atan2(dy, dx);
  FIG8_X[i] = x;
  FIG8_Y[i] = y;
  FIG8_YAWS[i] = yaw;
  FIG8_TX[i] = Math.cos(yaw);
  FIG8_TY[i] = Math.sin(yaw);
  FIG8_KAPPAS[i] = (dx * ddy - dy * ddx) / Math.pow(Math.max(1e-6, speedDs), 3);
  ARC_SEG[i] = speedDs * DS;
}

function buildLookaheadIndexTable(distM) {
  const offsets = new Int32Array(N_FIG8);
  for (let i = 0; i < N_FIG8; i++) {
    let acc = 0.0;
    let step = 0;
    while (acc < distM && step < (N_FIG8 >> 1)) {
      acc += ARC_SEG[(i + step) % N_FIG8];
      step += 1;
    }
    offsets[i] = (i + step) % N_FIG8;
  }
  return offsets;
}

const FIG8_LOOK_1 = buildLookaheadIndexTable(1.1);
const FIG8_LOOK_2 = buildLookaheadIndexTable(2.3);
const FIG8_LOOK_3 = buildLookaheadIndexTable(3.7);
const FIG8_LOOK_4 = buildLookaheadIndexTable(5.5);

export function wrapPi(angle) {
  const twoPi = 2.0 * Math.PI;
  let a = (angle + Math.PI) % twoPi;
  if (a < 0.0) a += twoPi;
  return a - Math.PI;
}

export function matchFig8Index(px, py, yaw, prevIdx = null, window = 38) {
  if (prevIdx === null) {
    let minD2 = Infinity;
    let bestD2Idx = 0;
    let bestCost = Infinity;
    let bestCostIdx = 0;
    for (let i = 0; i < N_FIG8; i++) {
      const dx = FIG8_X[i] - px;
      const dy = FIG8_Y[i] - py;
      const d2 = dx * dx + dy * dy;
      if (d2 < minD2) {
        minD2 = d2;
        bestD2Idx = i;
      }
      const cost = d2 + 0.8 * (1.0 - Math.cos(FIG8_YAWS[i] - yaw));
      if (cost < bestCost) {
        bestCost = cost;
        bestCostIdx = i;
      }
    }
    return minD2 > 4.0 ? bestD2Idx : bestCostIdx;
  }

  let bestIdx = prevIdx;
  let bestCost = Infinity;
  let bestD2 = Infinity;
  for (let off = -window; off <= window; off++) {
    const idx = (((prevIdx + off) % N_FIG8) + N_FIG8) % N_FIG8;
    const dx = FIG8_X[idx] - px;
    const dy = FIG8_Y[idx] - py;
    const d2 = dx * dx + dy * dy;
    const cost = d2 + 0.45 * (1.0 - Math.cos(FIG8_YAWS[idx] - yaw));
    if (cost < bestCost) {
      bestCost = cost;
      bestIdx = idx;
      bestD2 = d2;
    }
  }
  if (bestD2 > 9.0) {
    return matchFig8Index(px, py, yaw, null);
  }
  return bestIdx;
}

export function fig8Features(px, py, yaw, prevIdx = null) {
  const idx = matchFig8Index(px, py, yaw, prevIdx);
  const dx = px - FIG8_X[idx];
  const dy = py - FIG8_Y[idx];
  const eLat = -dx * FIG8_TY[idx] + dy * FIG8_TX[idx];
  const ePsi = wrapPi(yaw - FIG8_YAWS[idx]);
  const k0 = FIG8_KAPPAS[idx];
  const k1 = FIG8_KAPPAS[FIG8_LOOK_1[idx]];
  const k2 = FIG8_KAPPAS[FIG8_LOOK_2[idx]];
  const k3 = FIG8_KAPPAS[FIG8_LOOK_3[idx]];
  const k4 = FIG8_KAPPAS[FIG8_LOOK_4[idx]];
  return { idx, eLat, ePsi, k0, k1, k2, k3, k4 };
}

export function circleFeatures(px, py, yaw, direction = 1.0, radius = CIRCLE_RADIUS) {
  const r = Math.max(1e-4, Math.hypot(px, py));
  const radialAng = Math.atan2(py, px);
  const tangentYaw = radialAng + direction * 0.5 * Math.PI;
  const eLat = direction >= 0.0 ? radius - r : r - radius;
  const ePsi = wrapPi(yaw - tangentYaw);
  const k = direction / radius;
  return { eLat, ePsi, k0: k, k1: k, k2: k, k3: k };
}

export function cruiseFeatures(px, py, yaw, anchorX, anchorY, anchorYaw) {
  const tx = Math.cos(anchorYaw);
  const ty = Math.sin(anchorYaw);
  const dx = px - anchorX;
  const dy = py - anchorY;
  const eLat = -dx * ty + dy * tx;
  const ePsi = wrapPi(yaw - anchorYaw);
  return { eLat, ePsi, k0: 0.0, k1: 0.0, k2: 0.0, k3: 0.0 };
}

export function safeCornerSpeed(targetSpeed, k0, k1, k2, k3, k4 = 0.0, maxLatAccel = 4.4) {
  const maxK = Math.max(Math.abs(k0), Math.abs(k1), Math.abs(k2), Math.abs(k3), Math.abs(k4));
  if (maxK < 1e-4) return targetSpeed;
  const vCurveCap = Math.sqrt(maxLatAccel / maxK);
  return Math.max(1.8, Math.min(targetSpeed, vCurveCap));
}
