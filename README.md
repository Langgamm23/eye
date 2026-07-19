# EYE — AI Exam Proctor (Web Edition)

This is the same detection logic as the original desktop app (`eye.py`),
now running as a local web app: a Flask backend does the actual YOLO /
MediaPipe detection, and any browser on the machine (or on the same
Wi-Fi/LAN) can open it as a page instead of installing a PyQt6 app.

**Important — this still needs to run on a real computer with a
camera.** It cannot be hosted as a public website, the way a normal
page can: it needs a live webcam feed and runs real machine-learning
models locally. Think of it as "the desktop app, but you open it in a
browser" rather than "put it online."

## 1. Install (one-time, on the invigilator's computer)

```bash
cd eye_web
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will also auto-download the YOLO26 weights (`yolo26n.pt` /
`yolo26n-pose.pt`, or the `s` variant if you have an NVIDIA GPU) into
this folder, and MediaPipe's face/hand landmark models, the same way
the original desktop app did.

## 2. Run it

```bash
python server.py
```

You'll see:

```
EYE web is ready → open http://localhost:5000 in a browser on this machine
```

Open that link in Chrome/Edge/Firefox **on the same computer the
camera is attached to**. Click **Enable Camera**, allow the browser's
camera permission prompt, then **Start Monitoring**.

To let a teacher view it from another device on the same room's
Wi-Fi (e.g. a tablet at the back of the room), find this computer's
local IP address (e.g. `192.168.1.42`) and open
`http://192.168.1.42:5000` from that device instead of `localhost`.

## 3. What's in this folder

| File | What it is |
|---|---|
| `server.py` | Flask backend — model loading, per-frame detection, violation logic, evidence saving, SQLite persistence |
| `detection_engine.py` | Unchanged — your original computer-vision logic (SeatRegistry, GazeTracker, paper hand-off tracking, etc.) |
| `database.py` | Unchanged — your original SQLite layer |
| `static/index.html` + `app.js` + `style.css` | The monitoring page (webcam view, live alerts, settings) |
| `static/evidence.html` + `evidence.js` | The evidence gallery (browse/delete flagged screenshots) |
| `evidence/` | Saved violation screenshots (created automatically) |
| `proctor_system.db` | SQLite database (created automatically on first run) |

## 4. Notes for whoever is presenting this

- Everything that made the desktop version accurate — SeatRegistry's
  45-second seat-rebind window, the calibration period before
  OUT_OF_SEAT/COLLAB can fire, hysteresis-smoothed head-turn detection,
  confirmed (not just "nearby") paper hand-offs, per-type cooldowns —
  is carried over exactly, because it's the same `detection_engine.py`
  file and the same threshold constants from `eye.py`, just called
  from Flask instead of a Qt thread.
- Smartwatch detection stays disabled until a custom-trained watch
  model is dropped in `models/smartwatch.pt`, same as before — see
  `SmartwatchModelLoader`'s docstring in `detection_engine.py`.
- Performance presets (Max Speed / Balanced / Max Accuracy) work the
  same way; pick Max Accuracy for a big room or a camera far from the
  students.
