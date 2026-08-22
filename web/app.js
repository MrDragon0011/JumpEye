const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const goalInput = document.getElementById("goalInput");
const countDisplay = document.getElementById("countDisplay");
const countLabel = document.getElementById("countLabel");
const progressFill = document.getElementById("progressFill");
const statusLine = document.getElementById("statusLine");
const goalOverlay = document.getElementById("goalOverlay");
const modeSkeletonBtn = document.getElementById("modeSkeleton");
const modeBoxBtn = document.getElementById("modeBox");
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValue = document.getElementById("sensitivityValue");

const ADJACENT_PAIRS = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
];

let detector = null;
let running = false;
let rafId = null;

let jumpCount = 0;
let goalReached = false;
let state = "ground"; // "ground" | "airborne"
let baselineY = null;
let minYThisRep = null;
let history = [];
const HISTORY_LEN = 5;

let ankleBaselineY = null;
let ankleMinYThisRep = null;
let ankleHistory = [];
let ankleSeenThisRep = 0;

// Hip-to-ankle distance, in pixels, smoothed while standing on the ground.
// Using this (instead of a fixed fraction of the frame) as the reference for
// the jump threshold keeps the threshold physically meaningful — a fraction
// of the person's own leg length — regardless of how close they stand to the
// camera. A fixed frame-height fraction doesn't scale with the subject's
// apparent size, so standing closer to the camera made small real movements
// (e.g. rising onto tiptoes) cross the same pixel threshold a real jump
// would.
let legLengthPx = null;

let displayMode = "skeleton"; // "skeleton" | "box"
let sensitivity = parseFloat(sensitivitySlider.value);

function keypointMap(keypoints) {
  const map = {};
  for (const kp of keypoints) map[kp.name] = kp;
  return map;
}

function resizeCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return false;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  return true;
}

function waitForVideoReady() {
  return new Promise((resolve) => {
    if (resizeCanvasToVideo()) {
      resolve();
      return;
    }
    video.addEventListener(
      "loadedmetadata",
      () => {
        resizeCanvasToVideo();
        resolve();
      },
      { once: true }
    );
  });
}

const HAND_JOINTS = [
  { wrist: "left_wrist", elbow: "left_elbow" },
  { wrist: "right_wrist", elbow: "right_elbow" },
];

function drawHand(wrist, elbow) {
  // MoveNet has no finger keypoints, so a "hand" is rendered as a small
  // fanned splay of stub lines projecting past the wrist, in the direction
  // the forearm is already pointing.
  let dx = wrist.x - elbow.x;
  let dy = wrist.y - elbow.y;
  const len = Math.hypot(dx, dy) || 1;
  dx /= len;
  dy /= len;
  const perpX = -dy;
  const perpY = dx;
  const fingerLen = 14;
  const spread = 6;

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2.5;
  for (let i = -1; i <= 1; i++) {
    const baseX = wrist.x + perpX * spread * i * 0.6;
    const baseY = wrist.y + perpY * spread * i * 0.6;
    const tipX = baseX + dx * fingerLen + perpX * spread * i * 0.4;
    const tipY = baseY + dy * fingerLen + perpY * spread * i * 0.4;
    ctx.beginPath();
    ctx.moveTo(baseX, baseY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
  }

  ctx.fillStyle = "#fbbf24";
  ctx.beginPath();
  ctx.arc(wrist.x, wrist.y, 6, 0, 2 * Math.PI);
  ctx.fill();
}

function drawSkeleton(kpMap) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#5eead4";
  for (const [a, b] of ADJACENT_PAIRS) {
    const p1 = kpMap[a];
    const p2 = kpMap[b];
    if (p1 && p2 && p1.score > 0.3 && p2.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }
  ctx.fillStyle = "#fbbf24";
  for (const name in kpMap) {
    const kp = kpMap[name];
    if (kp.score > 0.3) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  for (const { wrist, elbow } of HAND_JOINTS) {
    const w = kpMap[wrist];
    const e = kpMap[elbow];
    if (w && e && w.score > 0.3 && e.score > 0.3) {
      drawHand(w, e);
    }
  }
}

function drawBoundingBox(keypoints) {
  const visible = keypoints.filter((k) => k.score > 0.3);
  if (visible.length === 0) return;
  const xs = visible.map((k) => k.x);
  const ys = visible.map((k) => k.y);
  const pad = 20;
  const x1 = Math.max(0, Math.min(...xs) - pad);
  const y1 = Math.max(0, Math.min(...ys) - pad);
  const x2 = Math.min(overlay.width, Math.max(...xs) + pad);
  const y2 = Math.min(overlay.height, Math.max(...ys) + pad);
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 3;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

function updateCountUI() {
  countDisplay.textContent = jumpCount;
  const goal = getGoal();
  const pct = goal > 0 ? Math.min(100, (jumpCount / goal) * 100) : 0;
  progressFill.style.width = `${pct}%`;

  if (jumpCount >= goal && goal > 0) {
    if (!goalReached) {
      goalReached = true;
      countDisplay.classList.add("goal-met");
      progressFill.classList.add("goal-met");
      countLabel.textContent = "goal reached!";
      goalOverlay.classList.add("show");
    }
  } else {
    if (goalReached) {
      goalReached = false;
      countDisplay.classList.remove("goal-met");
      progressFill.classList.remove("goal-met");
      countLabel.textContent = "jumps";
      goalOverlay.classList.remove("show");
    }
  }
}

function getGoal() {
  const v = parseInt(goalInput.value, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
}

function resetCounter() {
  jumpCount = 0;
  state = "ground";
  baselineY = null;
  minYThisRep = null;
  history = [];
  ankleBaselineY = null;
  ankleMinYThisRep = null;
  ankleHistory = [];
  ankleSeenThisRep = 0;
  legLengthPx = null;
  goalReached = false;
  countDisplay.classList.remove("goal-met");
  progressFill.classList.remove("goal-met");
  countLabel.textContent = "jumps";
  goalOverlay.classList.remove("show");
  updateCountUI();
}

function processHipY(hipY, ankleY) {
  history.push(hipY);
  if (history.length > HISTORY_LEN) history.shift();
  const smoothedY = history.reduce((a, b) => a + b, 0) / history.length;

  let smoothedAnkleY = null;
  if (ankleY !== null) {
    ankleHistory.push(ankleY);
    if (ankleHistory.length > HISTORY_LEN) ankleHistory.shift();
    smoothedAnkleY = ankleHistory.reduce((a, b) => a + b, 0) / ankleHistory.length;
  } else {
    ankleHistory = [];
  }

  if (baselineY === null) baselineY = smoothedY;

  // Track hip-to-ankle distance while grounded — the reference the jump
  // threshold scales against. Falls back to a typical body-proportion
  // estimate (ankles ~35% of frame height below hips) until the ankles have
  // been seen, or if they never are. Smoothed slowly and with outlier
  // rejection so a single noisy ankle-keypoint frame (motion blur, brief
  // occlusion) can't yank the threshold around mid-session — that jitter is
  // what made jump detection feel unstable.
  if (state === "ground" && smoothedAnkleY !== null) {
    const instantLegLength = smoothedAnkleY - smoothedY;
    const isOutlier =
      legLengthPx !== null && Math.abs(instantLegLength - legLengthPx) > legLengthPx * 0.3;
    if (instantLegLength > 0 && !isOutlier) {
      legLengthPx =
        legLengthPx === null ? instantLegLength : 0.97 * legLengthPx + 0.03 * instantLegLength;
    }
  }
  const legLengthRef = legLengthPx !== null ? legLengthPx : overlay.height * 0.35;

  const thresholdPx = sensitivity * legLengthRef;
  const noiseBand = 0.01 * overlay.height;
  // Ankles travel less on screen than hips during a real jump (shorter lever
  // from the ground), so corroboration only needs a fraction of the hip
  // threshold — just enough to rule out an arms-only motion, where the feet
  // stay planted and this stays ~0.
  const ankleThresholdPx = thresholdPx * 0.35;

  if (state === "ground") {
    baselineY = 0.9 * baselineY + 0.1 * smoothedY;
    if (smoothedAnkleY !== null) {
      ankleBaselineY =
        ankleBaselineY === null ? smoothedAnkleY : 0.9 * ankleBaselineY + 0.1 * smoothedAnkleY;
    }
    if (smoothedY < baselineY - thresholdPx) {
      state = "airborne";
      minYThisRep = smoothedY;
      ankleMinYThisRep = smoothedAnkleY;
      ankleSeenThisRep = smoothedAnkleY !== null ? 1 : 0;
    }
  } else {
    minYThisRep = Math.min(minYThisRep, smoothedY);
    if (smoothedAnkleY !== null) {
      ankleSeenThisRep += 1;
      ankleMinYThisRep =
        ankleMinYThisRep === null ? smoothedAnkleY : Math.min(ankleMinYThisRep, smoothedAnkleY);
    }
    if (smoothedY > baselineY - noiseBand) {
      const peakHeight = baselineY - minYThisRep;
      // Only trust ankle corroboration if it was tracked for several frames
      // of the rep, not just a single noisy sample — a lone glimpse of the
      // ankles (motion blur, brief occlusion) shouldn't be able to veto an
      // otherwise-clear jump.
      const ankleReliable = ankleSeenThisRep >= 3 && ankleBaselineY !== null && ankleMinYThisRep !== null;
      const anklePeakHeight = ankleReliable ? ankleBaselineY - ankleMinYThisRep : null;
      // If the ankles were reliably tracked through the rep, require them to
      // have moved too — filters out arm-only motion, which only shifts the
      // hip keypoint estimate without actually lifting the body. Otherwise
      // fall back to hip-only.
      const ankleConfirms = anklePeakHeight === null || anklePeakHeight >= ankleThresholdPx;
      if (peakHeight >= thresholdPx && ankleConfirms) {
        jumpCount += 1;
        updateCountUI();
      }
      // Snap the baseline to the landing position instead of letting it
      // crawl back via slow EMA — landing IS the new "standing" reference.
      // Use the raw (unsmoothed) current values and reseed the moving-average
      // windows, rather than `smoothedY`/`smoothedAnkleY` — those are still a
      // blend of peak and landing samples for a few frames after a sharp
      // transition, so snapping to them left the baseline still skewed and
      // made consecutive jumps (little standing time between reps)
      // progressively harder to register.
      baselineY = hipY;
      history = [hipY];
      if (ankleY !== null) {
        ankleBaselineY = ankleY;
        ankleHistory = [ankleY];
      }
      state = "ground";
    }
  }
}

async function detectLoop() {
  if (!running) return;
  const poses = await detector.estimatePoses(video, { flipHorizontal: false });

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (poses.length > 0) {
    const kpMap = keypointMap(poses[0].keypoints);
    const leftHip = kpMap["left_hip"];
    const rightHip = kpMap["right_hip"];
    const nose = kpMap["nose"];
    const leftAnkle = kpMap["left_ankle"];
    const rightAnkle = kpMap["right_ankle"];

    let hipY = null;
    if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
      hipY = (leftHip.y + rightHip.y) / 2;
    } else if (nose && nose.score > 0.3) {
      hipY = nose.y;
    }

    let ankleY = null;
    if (leftAnkle && rightAnkle && leftAnkle.score > 0.3 && rightAnkle.score > 0.3) {
      ankleY = (leftAnkle.y + rightAnkle.y) / 2;
    }

    if (displayMode === "skeleton") {
      drawSkeleton(kpMap);
    } else {
      drawBoundingBox(poses[0].keypoints);
    }

    if (hipY !== null) {
      processHipY(hipY, ankleY);
      statusLine.textContent = `Tracking · state: ${state}`;
    } else {
      history = [];
      ankleHistory = [];
      statusLine.textContent = "Person detected but key joints not visible";
    }
  } else {
    history = [];
    ankleHistory = [];
    statusLine.textContent = "No person detected";
  }

  rafId = requestAnimationFrame(detectLoop);
}

async function start() {
  startBtn.disabled = true;
  statusLine.textContent = "Requesting camera...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await waitForVideoReady();
  } catch (err) {
    statusLine.textContent = "Camera access denied or unavailable.";
    startBtn.disabled = false;
    return;
  }

  if (!detector) {
    statusLine.textContent = "Loading pose model...";
    try {
      await tf.setBackend("webgl");
    } catch (e) {
      await tf.setBackend("cpu");
    }
    await tf.ready();
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
  }

  running = true;
  startBtn.textContent = "Stop camera";
  startBtn.disabled = false;
  statusLine.textContent = "Tracking...";
  detectLoop();
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.textContent = "Start camera";
  statusLine.textContent = "Camera stopped.";
}

startBtn.addEventListener("click", () => {
  if (running) {
    stop();
  } else {
    start();
  }
});

resetBtn.addEventListener("click", resetCounter);

modeSkeletonBtn.addEventListener("click", () => {
  displayMode = "skeleton";
  modeSkeletonBtn.classList.add("active");
  modeBoxBtn.classList.remove("active");
});

modeBoxBtn.addEventListener("click", () => {
  displayMode = "box";
  modeBoxBtn.classList.add("active");
  modeSkeletonBtn.classList.remove("active");
});

sensitivitySlider.addEventListener("input", () => {
  sensitivity = parseFloat(sensitivitySlider.value);
  sensitivityValue.textContent = sensitivity.toFixed(2);
});

goalInput.addEventListener("input", updateCountUI);

updateCountUI();
