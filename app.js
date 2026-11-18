// Browser entry point wiring @mujoco/mujoco 4.0.1 WASM, Three.js viewer, and UI controls.

import loadMujoco from '@mujoco/mujoco';
import { PelicanPolicy } from './policy.js';
import { PelicanSimController } from './controller.js';
import { PelicanThreeViewer } from './viewer.js';

const MUJOCO_CDN_BASE = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.1/';

const ui = {
  overlay: document.getElementById('loading-overlay'),
  loadStatus: document.getElementById('loading-status'),
  loadBar: document.getElementById('loading-bar'),
  panel: document.getElementById('control-panel'),
  pauseBtn: document.getElementById('pause-btn'),
  collapseBtn: document.getElementById('collapse-btn'),
  ctrlLabel: document.getElementById('controller-label'),
  rideMode: document.getElementById('ride-mode'),
  targetSpeed: document.getElementById('target-speed'),
  targetSpeedVal: document.getElementById('target-speed-val'),
  manualSteerLbl: document.getElementById('manual-steer-lbl'),
  manualSteerWrap: document.getElementById('manual-steer-wrap'),
  manualSteer: document.getElementById('manual-steer'),
  manualSteerVal: document.getElementById('manual-steer-val'),
  wingMode: document.getElementById('wing-mode'),
  flapAmpLbl: document.getElementById('flap-amp-lbl'),
  flapAmpWrap: document.getElementById('flap-amp-wrap'),
  flapAmp: document.getElementById('flap-amp'),
  flapAmpVal: document.getElementById('flap-amp-val'),
  envMode: document.getElementById('env-mode'),
  usePolicy: document.getElementById('use-policy'),
  beakHold: document.getElementById('beak-hold'),
  headLook: document.getElementById('head-look'),
  followCam: document.getElementById('follow-cam'),
  telStatus: document.getElementById('tel-status'),
  telMetrics: document.getElementById('tel-metrics'),
  telSpeed: document.getElementById('tel-speed'),
  telAngles: document.getElementById('tel-angles'),
  chart: document.getElementById('telemetry-chart'),
};

const chartCtx = ui.chart.getContext('2d');
const MAX_HIST = 240;
let histT = [];
let histLean = [];
let histPitch = [];
let histSpeed4 = [];
let lastHistTime = -1.0;

function clearChartHistory() {
  histT = [];
  histLean = [];
  histPitch = [];
  histSpeed4 = [];
  lastHistTime = -1.0;
}

function drawTelemetryChart() {
  const w = ui.chart.width;
  const h = ui.chart.height;
  chartCtx.clearRect(0, 0, w, h);

  let peakY = 30.0;
  for (let i = 0; i < histT.length; i++) {
    peakY = Math.max(peakY, Math.abs(histLean[i]), Math.abs(histPitch[i]), Math.abs(histSpeed4[i]));
  }
  const yLim = Math.min(110.0, peakY + 6.0);

  // Zero baseline
  chartCtx.strokeStyle = 'rgba(255,255,255,0.14)';
  chartCtx.lineWidth = 1;
  chartCtx.beginPath();
  chartCtx.moveTo(0, h * 0.5);
  chartCtx.lineTo(w, h * 0.5);
  chartCtx.stroke();

  if (histT.length < 2) return;
  const t0 = histT[0];
  const span = Math.max(1.0, histT[histT.length - 1] - t0);

  const plotSeries = (arr, color) => {
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 1.6;
    chartCtx.beginPath();
    for (let i = 0; i < histT.length; i++) {
      const x = ((histT[i] - t0) / span) * w;
      const y = h * 0.5 - (arr[i] / yLim) * (h * 0.44);
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
  };

  plotSeries(histLean, '#63b3ed');
  plotSeries(histPitch, '#f6ad55');
  plotSeries(histSpeed4, '#68d391');
}

function setProgress(frac, text) {
  ui.loadBar.style.width = `${Math.round(frac * 100)}%`;
  if (text) ui.loadStatus.textContent = text;
}

async function main() {
  setProgress(0.05, 'Initializing MuJoCo 4.0.1 WebAssembly...');
  const [mujoco, policyRes, manifestRes] = await Promise.all([
    loadMujoco({ locateFile: (path) => MUJOCO_CDN_BASE + path }),
    fetch('trained_policy.json').then((r) => r.json()),
    fetch('asset_manifest.json').then((r) => r.json()),
  ]);

  mujoco.FS.mkdirTree('/working/assets/pelican_bicycle/meshes');
  mujoco.FS.mkdirTree('/working/assets/pelican_bicycle/textures');

  const sceneNames = ['coastal_road', 'slalom_cones', 'checker'];
  const sceneXmlMap = {};

  const totalAssets = sceneNames.length + manifestRes.meshes.length + manifestRes.textures.length;
  let loadedCount = 0;
  const bump = () => {
    loadedCount += 1;
    setProgress(0.15 + 0.80 * (loadedCount / totalAssets), `Loading 3D meshes & PBR textures (${loadedCount}/${totalAssets})...`);
  };

  await Promise.all([
    ...sceneNames.map(async (name) => {
      const text = await fetch(`scenes/${name}.xml`).then((r) => r.text());
      sceneXmlMap[name] = text;
      bump();
    }),
    ...manifestRes.meshes.map(async (fn) => {
      const buf = new Uint8Array(await fetch(`assets/pelican_bicycle/meshes/${fn}`).then((r) => r.arrayBuffer()));
      mujoco.FS.writeFile(`/working/assets/pelican_bicycle/meshes/${fn}`, buf);
      bump();
    }),
    ...manifestRes.textures.map(async (fn) => {
      const buf = new Uint8Array(await fetch(`assets/pelican_bicycle/textures/${fn}`).then((r) => r.arrayBuffer()));
      mujoco.FS.writeFile(`/working/assets/pelican_bicycle/textures/${fn}`, buf);
      bump();
    }),
  ]);

  setProgress(0.97, 'Compiling MuJoCo scene & starting simulation...');

  const policy = new PelicanPolicy(policyRes);
  const controller = new PelicanSimController(policy);
  const viewer = new PelicanThreeViewer(document.getElementById('viewport'));

  let model = null;
  let data = null;
  let paused = false;

  function loadEnvironment(envName) {
    if (data) data.delete();
    if (model) model.delete();
    mujoco.FS.writeFile('/working/scene.xml', sceneXmlMap[envName]);
    model = mujoco.MjModel.mj_loadXML('/working/scene.xml');
    data = new mujoco.MjData(model);
    controller.bindModel(mujoco, model, data);
    controller.initializeRollingState();
    viewer.buildSceneFromModel(mujoco, model, data);
    clearChartHistory();

    // Preserve active camera choice if fixed
    const activeCamBtn = document.querySelector('.cam-btn.active');
    if (activeCamBtn) {
      viewer.setCameraView(activeCamBtn.dataset.cam);
    }
  }

  loadEnvironment(ui.envMode.value);
  setProgress(1.0, 'Ready!');
  ui.overlay.classList.add('hidden');

  // Wire UI controls
  ui.collapseBtn.addEventListener('click', () => {
    const collapsed = ui.panel.classList.toggle('collapsed');
    ui.collapseBtn.textContent = collapsed ? 'Show' : 'Hide';
  });

  ui.pauseBtn.addEventListener('click', () => {
    paused = !paused;
    ui.pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  });

  ui.rideMode.addEventListener('change', () => {
    const m = ui.rideMode.value;
    controller.setMode(m);
    const isManual = m === 'manual';
    ui.manualSteerLbl.classList.toggle('hidden', !isManual);
    ui.manualSteerWrap.classList.toggle('hidden', !isManual);
  });

  ui.targetSpeed.addEventListener('input', () => {
    const v = parseFloat(ui.targetSpeed.value);
    controller.targetSpeed = v;
    ui.targetSpeedVal.textContent = `${v.toFixed(2)} m/s`;
  });

  ui.manualSteer.addEventListener('input', () => {
    const v = parseFloat(ui.manualSteer.value);
    controller.manualSteer = v;
    ui.manualSteerVal.textContent = v.toFixed(2);
  });

  ui.wingMode.addEventListener('change', () => {
    const wm = ui.wingMode.value;
    controller.wingMode = wm;
    const isFlap = wm === 'flap';
    ui.flapAmpLbl.classList.toggle('hidden', !isFlap);
    ui.flapAmpWrap.classList.toggle('hidden', !isFlap);
  });

  ui.flapAmp.addEventListener('input', () => {
    const v = parseFloat(ui.flapAmp.value);
    controller.flapIntensity = v;
    ui.flapAmpVal.textContent = v.toFixed(2);
  });

  ui.envMode.addEventListener('change', () => {
    loadEnvironment(ui.envMode.value);
  });

  ui.usePolicy.addEventListener('change', () => {
    controller.useLearnedPolicy = ui.usePolicy.checked;
  });

  ui.beakHold.addEventListener('change', () => {
    controller.beakOpenHold = ui.beakHold.checked;
  });

  ui.headLook.addEventListener('change', () => {
    controller.autoHeadLook = ui.headLook.checked;
  });

  ui.followCam.addEventListener('change', () => {
    viewer.followCamera = ui.followCam.checked;
  });

  document.querySelectorAll('[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const act = btn.dataset.action;
      controller.triggerAction(act);
      if (act === 'reset_ride') {
        clearChartHistory();
        viewer.syncTransforms();
        if (viewer.cameraMode === 'orbit') viewer.resetOrbitCamera();
      }
    });
  });

  document.querySelectorAll('.cam-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cam-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const cam = btn.dataset.cam;
      ui.followCam.checked = cam === 'orbit';
      viewer.followCamera = ui.followCam.checked;
      viewer.setCameraView(cam);
    });
  });

  function updateTelemetryUI() {
    const t = controller.getTelemetry();
    ui.ctrlLabel.textContent = `Controller: ${t.policyLabel}`;

    let statusCls = 'status-line';
    let statusTxt = 'UPRIGHT';
    if (!t.upright) {
      statusCls = 'status-line tipped';
      statusTxt = 'TIPPED (click Reset Ride)';
    } else if (t.airborne) {
      statusCls = 'status-line airborne';
      statusTxt = 'AIRBORNE!';
    }
    const trickStr = t.activeTrick ? ` | Trick: ${t.activeTrick}` : '';
    const nextStr = t.pendingTrick ? ` | Next: ${t.pendingTrick}` : '';
    ui.telStatus.className = statusCls;
    ui.telStatus.textContent = `Status: ${statusTxt}${trickStr}${nextStr} | Tricks: ${t.trickCount}`;

    document.querySelectorAll('.trick-btn').forEach((btn) => {
      const lbl = btn.textContent.replace('!', '').toUpperCase().replace('-', ' ');
      const activeLbl = t.activeTrick.replace('-', ' ');
      btn.classList.toggle('active', activeLbl !== '' && activeLbl.includes(lbl.split(' ')[0]));
    });

    ui.telMetrics.textContent = `t = ${t.simTime.toFixed(1)} s | Dist = ${t.distanceM.toFixed(1)} m | Err = ${t.roadErrM.toFixed(2)} m`;
    ui.telSpeed.textContent = `Speed: ${t.speedMps.toFixed(2)} m/s | Cadence: ${Math.round(t.crankRpm)} rpm`;
    ui.telAngles.textContent = `Lean: ${t.leanDeg >= 0 ? '+' : ''}${t.leanDeg.toFixed(1)}° | Pitch: ${t.pitchDeg >= 0 ? '+' : ''}${t.pitchDeg.toFixed(1)}° | Steer: ${t.steerDeg >= 0 ? '+' : ''}${t.steerDeg.toFixed(1)}°`;

    if (t.simTime < lastHistTime) {
      clearChartHistory();
    }
    if (t.simTime - lastHistTime >= 0.04) {
      histT.push(t.simTime);
      histLean.push(t.leanDeg);
      histPitch.push(t.pitchDeg);
      histSpeed4.push(t.speedMps * 4.0);
      if (histT.length > MAX_HIST) {
        histT.shift();
        histLean.shift();
        histPitch.shift();
        histSpeed4.shift();
      }
      lastHistTime = t.simTime;
      drawTelemetryChart();
    }
  }

  let accum = 0.0;
  let lastNow = performance.now();
  let frameCounter = 0;

  function animate(now) {
    requestAnimationFrame(animate);
    const dtFrame = Math.min(0.05, Math.max(0.0, (now - lastNow) * 0.001));
    lastNow = now;

    if (!paused && model && data) {
      accum += dtFrame;
      const stepDt = 0.002;
      let steps = 0;
      while (accum >= stepDt && steps < 25) {
        controller.preStep();
        mujoco.mj_step(model, data);
        controller.postStep();
        accum -= stepDt;
        steps += 1;
      }
      if (accum > stepDt * 5) accum = 0.0;
    }

    viewer.syncTransforms();
    viewer.render();

    frameCounter += 1;
    if (frameCounter % 3 === 0) {
      updateTelemetryUI();
    }
  }

  // Expose lightweight debug hook for automated browser checks (read-only state inspection)
  window.__pelicanSim = { controller, getData: () => data, getModel: () => model };
  requestAnimationFrame(animate);
}

main().catch((err) => {
  console.error(err);
  setProgress(1.0, `Failed to load simulation: ${err.message}`);
});
