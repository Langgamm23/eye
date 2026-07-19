const CAPTURE_W = 1280, CAPTURE_H = 720;

const video   = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx    = overlay.getContext('2d');
const flash   = document.getElementById('flash');
const logBox  = document.getElementById('log');

const capCanvas = document.createElement('canvas');
capCanvas.width = CAPTURE_W;
capCanvas.height = CAPTURE_H;
const cctx = capCanvas.getContext('2d');

let stream = null;
let running = false;
let fpsCounter = 0, fpsTimer = performance.now();
let totalViolations = 0;

function log(msg) {
  const line = document.createElement('div');
  const t = new Date().toLocaleTimeString();
  line.textContent = `[${t}] ${msg}`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

// ── camera ──────────────────────────────────────────────────────────
document.getElementById('btnCamera').addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: CAPTURE_W }, height: { ideal: CAPTURE_H } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    document.getElementById('camHint').textContent = 'Camera ready. Click Start Monitoring.';
    log('Camera enabled.');
  } catch (e) {
    document.getElementById('camHint').textContent = 'Camera access failed: ' + e.message;
    log('Camera error: ' + e.message);
  }
});

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}
window.addEventListener('resize', resizeOverlay);

// ── settings ────────────────────────────────────────────────────────
async function pushSettings() {
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      preset: document.getElementById('preset').value,
      checks: {
        phone: document.getElementById('chkPhone').checked,
        book: document.getElementById('chkBook').checked,
        smartwatch: document.getElementById('chkWatch').checked,
        hand: document.getElementById('chkHand').checked,
      },
    }),
  });
}
['preset', 'chkPhone', 'chkBook', 'chkWatch', 'chkHand'].forEach(id => {
  document.getElementById(id).addEventListener('change', pushSettings);
});

// ── session control ─────────────────────────────────────────────────
document.getElementById('btnStart').addEventListener('click', async () => {
  if (!stream) {
    alert('Enable the camera first.');
    return;
  }
  await pushSettings();
  const name = document.getElementById('sessionName').value;
  const duration = document.getElementById('duration').value;
  const res = await fetch('/api/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, duration }),
  });
  const data = await res.json();
  log(`Session started (id ${data.session_id}).`);
  running = true;
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStop').disabled = false;
  requestAnimationFrame(loop);
});

document.getElementById('btnStop').addEventListener('click', async () => {
  running = false;
  await fetch('/api/session/end', { method: 'POST' });
  log('Session ended.');
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled = true;
});

// ── main frame loop ─────────────────────────────────────────────────
async function loop() {
  if (!running) return;
  const t0 = performance.now();

  cctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
  capCanvas.toBlob(async (blob) => {
    if (!running) return;
    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');
    try {
      const res = await fetch('/api/frame', { method: 'POST', body: form });
      const data = await res.json();
      if (!data.error) {
        render(data);
      }
    } catch (e) {
      // network hiccup on one frame — just try the next one
    }
    fpsCounter++;
    const now = performance.now();
    if (now - fpsTimer >= 500) {
      document.getElementById('statFps').textContent = (fpsCounter / ((now - fpsTimer) / 1000)).toFixed(1);
      fpsCounter = 0;
      fpsTimer = now;
    }
    if (running) requestAnimationFrame(loop);
  }, 'image/jpeg', 0.82);
}

function render(data) {
  resizeOverlay();
  const sx = overlay.width / CAPTURE_W;
  const sy = overlay.height / CAPTURE_H;
  octx.clearRect(0, 0, overlay.width, overlay.height);

  // seat boxes
  for (const s of data.seats) {
    const [x1, y1, x2, y2] = s.box;
    let colour = '#64c864';
    if (s.pct >= 75) colour = '#f0403c';
    else if (s.pct >= 40) colour = '#ffa500';
    octx.strokeStyle = colour;
    octx.lineWidth = 2;
    octx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    const label = s.calibrated ? `S${s.seat_id} · ${s.pct.toFixed(0)}%` : `S${s.seat_id} · calibrating…`;
    octx.font = '13px Segoe UI';
    const tw = octx.measureText(label).width;
    octx.fillStyle = colour;
    octx.fillRect(x1 * sx, Math.max(y1 * sy - 20, 0), tw + 10, 20);
    octx.fillStyle = '#000';
    octx.fillText(label, x1 * sx + 5, Math.max(y1 * sy - 5, 14));
  }

  // phones
  for (const [x1, y1, x2, y2, conf] of data.phones) {
    octx.strokeStyle = '#f03c3c';
    octx.lineWidth = 3;
    octx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    octx.fillStyle = '#f03c3c';
    octx.font = '12px Segoe UI';
    octx.fillText(`PHONE ${(conf * 100).toFixed(0)}%`, x1 * sx, Math.max(y1 * sy - 6, 12));
  }

  // paper-like objects
  for (const [x1, y1, x2, y2, ptype] of data.papers) {
    octx.strokeStyle = '#3c78f0';
    octx.lineWidth = 2;
    octx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    octx.fillStyle = '#3c78f0';
    octx.font = '12px Segoe UI';
    octx.fillText(ptype, x1 * sx, Math.max(y1 * sy - 6, 12));
  }

  document.getElementById('cardStudents').textContent = data.student_count;
  document.getElementById('statElapsed').textContent = data.exam_elapsed || '00:00:00';
  document.getElementById('detBar').textContent =
    `Tracking ${data.student_count} student(s) | ${data.phones.length} phone(s) | ${data.papers.length} paper item(s) detected`;

  if (data.violations && data.violations.length) {
    flash.classList.add('on');
    setTimeout(() => flash.classList.remove('on'), 150);
    for (const v of data.violations) {
      totalViolations++;
      addAlert(v);
      log(`⚠ Student ${v.student_id} — ${v.violation_type} · ${(v.confidence * 100).toFixed(0)}% conf. · ${v.exam_elapsed}`);
    }
    document.getElementById('cardViolations').textContent = totalViolations;
  }
}

function addAlert(v) {
  const el = document.createElement('div');
  el.className = 'alert-item';
  el.innerHTML = `<div class="a-head">S${v.student_id} — ${v.violation_type}</div>
                  <div class="a-sub">${(v.confidence * 100).toFixed(0)}% · ${v.exam_elapsed} · ${v.reason || ''}</div>`;
  const box = document.getElementById('alerts');
  box.insertBefore(el, box.firstChild);
  while (box.children.length > 25) box.removeChild(box.lastChild);
}

// ── boot status ─────────────────────────────────────────────────────
(async function boot() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('statBackend').textContent = data.device;
    document.getElementById('statTier').textContent = 'YOLO26' + data.model_tier.toUpperCase();
    log('Backend ready: ' + data.device);
  } catch (e) {
    log('Could not reach backend — is server.py running?');
  }
})();
