let violations = [];
let current = 0;

const VIOLATION_LABELS = {
  PHONE: 'Phone Usage', COPYING: 'Head Turning', PAPER: 'Paper Passing',
  OUT_OF_SEAT: 'Left Seat', COLLAB: 'Collaborating', HAND_SIGNAL: 'Hand Signaling',
  SMARTWATCH: 'Smartwatch (experimental)',
};

async function loadViolations() {
  const res = await fetch('/api/violations');
  violations = await res.json();
  if (current >= violations.length) current = Math.max(violations.length - 1, 0);
  renderGrid();
  renderCurrent();
}

function evidenceUrl(path) {
  if (!path) return null;
  const name = path.split(/[\\/]/).pop();
  return '/evidence/' + encodeURIComponent(name);
}

function renderCurrent() {
  const wrap = document.getElementById('imageWrap');
  const details = document.getElementById('details');
  const progress = document.getElementById('progress');

  if (!violations.length) {
    wrap.innerHTML = '<span style="color:#666;">No evidence to display</span>';
    details.innerHTML = '<div>Student ID: —</div><div>Violation: —</div><div>Wall clock: —</div><div>Exam elapsed: —</div>';
    document.getElementById('conf').textContent = 'Confidence: —';
    document.getElementById('reason').textContent = '';
    progress.textContent = '0 / 0';
    return;
  }

  const v = violations[current];
  const url = evidenceUrl(v.evidence_path);
  wrap.innerHTML = url ? `<img src="${url}" alt="evidence">` : '<span style="color:#666;">Image file not found</span>';

  const label = VIOLATION_LABELS[v.violation_type] || v.violation_type;
  let extra = '';
  if (v.role === 'sent') extra = `sent to Student ${v.partner_id}`;
  else if (v.role === 'received') extra = `received from Student ${v.partner_id}`;

  details.innerHTML = `
    <div>Student ID: ${v.student_id}</div>
    <div>Violation: ${label}</div>
    <div>Wall clock: ${v.datetime || '—'}</div>
    <div>Exam elapsed: ${v.exam_elapsed || '—'}</div>
  `;
  const conf = v.confidence;
  const confEl = document.getElementById('conf');
  if (conf !== null && conf !== undefined) {
    const pct = conf <= 1 ? conf * 100 : conf;
    confEl.textContent = `Confidence: ${pct.toFixed(0)}%`;
    confEl.style.color = pct >= 80 ? '#00e676' : (pct >= 55 ? '#ffa726' : '#ef5350');
  } else {
    confEl.textContent = 'Confidence: — (record predates confidence scoring)';
    confEl.style.color = '#666';
  }

  let reason = v.reason || '';
  if (extra) reason = reason ? `${reason} · ${extra}` : extra;
  if (v.paper_type) reason += (reason ? '  ·  ' : '') + `Paper type: ${v.paper_type}`;
  document.getElementById('reason').textContent = reason;

  progress.textContent = `${current + 1} / ${violations.length}`;
  document.getElementById('prevBtn').disabled = current === 0;
  document.getElementById('nextBtn').disabled = current >= violations.length - 1;

  document.querySelectorAll('.ev-thumb').forEach((el, i) => el.classList.toggle('active', i === current));
}

function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  violations.forEach((v, i) => {
    const div = document.createElement('div');
    div.className = 'ev-thumb' + (i === current ? ' active' : '');
    const url = evidenceUrl(v.evidence_path);
    div.innerHTML = url ? `<img src="${url}">` : '';
    div.title = `S${v.student_id} · ${v.violation_type}`;
    div.addEventListener('click', () => { current = i; renderCurrent(); });
    grid.appendChild(div);
  });
}

document.getElementById('prevBtn').addEventListener('click', () => {
  if (current > 0) { current--; renderCurrent(); }
});
document.getElementById('nextBtn').addEventListener('click', () => {
  if (current < violations.length - 1) { current++; renderCurrent(); }
});
document.getElementById('refreshBtn').addEventListener('click', loadViolations);
document.getElementById('removeBtn').addEventListener('click', async () => {
  if (!violations.length) return;
  const v = violations[current];
  if (!confirm('Delete this violation record?')) return;
  if (v.id) await fetch(`/api/violations/${v.id}`, { method: 'DELETE' });
  await loadViolations();
});

loadViolations();
