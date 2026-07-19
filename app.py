"""
server.py
──────────
EYE, ported from a PyQt6 desktop app to a local browser-based web app.

WHY THIS EXISTS
────────────────
The original eye.py is a PyQt6 QMainWindow: camera capture, drawing,
and the settings UI were all tied directly to Qt widgets and a QThread
worker. None of that can run in a browser. This file keeps the parts
that were already framework-agnostic (detection_engine.py, database.py
— neither imports PyQt6) and re-implements the orchestration layer
that used to live in eye.py's InferenceWorker + AIExamProctor classes
as plain Python, running inside a Flask process instead of a Qt event
loop. The browser sends webcam frames over HTTP; this returns the same
boxes/labels/violations the desktop app used to draw with OpenCV, and
the frontend (static/app.js) draws them on an HTML5 canvas instead.

WHAT DID NOT CHANGE
─────────────────────
- detection_engine.py and database.py are byte-for-byte your originals.
- Every threshold, cooldown, and heuristic below (PHONE_SEC, COPY_SEC,
  the calibration window, the accumulate-then-fire pattern, confidence
  scoring) is carried over unchanged from AIExamProctor in eye.py.

WHAT DID CHANGE
──────────────────
- No camera enumeration / QComboBox — the browser's own getUserMedia
  picks the camera, same as any website that asks for webcam access.
- One global ProctorEngine instance instead of one per QMainWindow —
  this process is meant to run on one teacher's machine for one
  classroom at a time, same as the desktop app was.
- Frames arrive as JPEG bytes over POST instead of a QThread mailbox;
  everything downstream (SeatRegistry, GazeTracker, the six _chk_*
  checks, _fire()/evidence saving) is the same logic, sequential
  per-request instead of running on a background QThread, because a
  browser request/response cycle is itself the frame loop now.

RUNNING THIS
──────────────
This is still a LOCAL app — it needs a real camera and runs real YOLO/
MediaPipe models on whatever machine it's started on, so it can't be
hosted as a public website the way a static page can. Run it on the
teacher's own computer, then open the printed http://localhost:5000
URL in a browser on that SAME machine (or any device on the same
Wi-Fi/LAN, using that machine's local IP instead of localhost).
"""
import os
import io
import math
import time
import statistics
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, Response
from ultralytics import YOLO

from database import Database
from detection_engine import (
    clamp, box_iou, classify_paper, SeatRegistry, PaperOwnershipTracker,
    HeadGazeEstimator, GazeTracker, score_phone_gesture, SeatFrame,
    FrameAnalysis, PhoneProximityScanner, merge_phone_detections,
    ActiveChecks, HandSignalDetector, HandRaiseTracker, detect_wrist_object,
    SmartwatchModelLoader,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(_SCRIPT_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

CAPTURE_W, CAPTURE_H = 1280, 720
CLS_PERSON, CLS_PHONE, CLS_BOOK = 0, 67, 73

PERF_PRESETS = {
    "Max Speed":     {"imgsz": 480, "conf": 0.42, "iou": 0.50, "max_det": 80,  "obj_skip": 3, "face_batch": 3,  "min_person_h_div": 22, "phone_scan_batch": 0, "smartwatch_batch": 5},
    "Balanced":      {"imgsz": 640, "conf": 0.35, "iou": 0.45, "max_det": 100, "obj_skip": 2, "face_batch": 6,  "min_person_h_div": 18, "phone_scan_batch": 3, "smartwatch_batch": 8},
    "Max Accuracy":  {"imgsz": 960, "conf": 0.28, "iou": 0.40, "max_det": 150, "obj_skip": 1, "face_batch": 10, "min_person_h_div": 26, "phone_scan_batch": 6, "smartwatch_batch": 14},
}
DEFAULT_PRESET = "Balanced"

VSTYLES = {
    "PHONE":       {"lbl": "Phone Usage",              "icon": "📱"},
    "COPYING":     {"lbl": "Head Turning",              "icon": "👀"},
    "PAPER":       {"lbl": "Paper Passing",             "icon": "📄"},
    "OUT_OF_SEAT": {"lbl": "Left Seat",                 "icon": "🚶"},
    "COLLAB":      {"lbl": "Collaborating",             "icon": "🗣️"},
    "SMARTWATCH":  {"lbl": "Smartwatch (experimental)", "icon": "⌚"},
    "HAND_SIGNAL": {"lbl": "Hand Signaling",            "icon": "🖐️"},
}
PAPER_STYLES = {
    "YELLOW_PAD":   {"lbl": "Yellow Pad"},
    "BUBBLE_SHEET": {"lbl": "Bubble Sheet"},
    "TEST_PAPER":   {"lbl": "Test Paper"},
    "NOTEBOOK":     {"lbl": "Notebook"},
}


def pick_device():
    try:
        if torch.cuda.is_available():
            return "cuda:0", True, f"CUDA · {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", False, "Apple MPS"
    except Exception:
        pass
    return "cpu", False, "CPU"


def resolve_model_path(base_name):
    stem = base_name.rsplit(".", 1)[0]
    for folder in (_SCRIPT_DIR, os.getcwd()):
        for candidate in (f"{stem}.engine", f"{stem}_openvino_model", f"{stem}.onnx", base_name):
            p = os.path.join(folder, candidate)
            if os.path.exists(p):
                return p
    return os.path.join(_SCRIPT_DIR, base_name)


# ─────────────────────────── the engine ────────────────────────────────

class ProctorEngine:
    """
    Everything eye.py's InferenceWorker + AIExamProctor used to hold as
    instance state, minus anything Qt-specific. One instance serves the
    whole app (one classroom / one camera at a time, same as before).
    """

    PHONE_SEC, COPY_SEC, COPY_WARNINGS, WARN_DECAY_SEC = 0.7, 1.4, 3, 240
    SEAT_SEC, COLLAB_SEC, WATCH_SEC, COOL_SEC = 1.8, 1.6, 5.0, 12
    CALIB_SEC, CALIB_MIN_SAMPLES = 2.5, 5

    def __init__(self):
        self.device, self.use_half, self.dev_label = pick_device()
        model_size = "s" if torch.cuda.is_available() else "n"
        self.det_model = YOLO(resolve_model_path(f"yolo26{model_size}.pt"))
        self.pose_model = YOLO(resolve_model_path(f"yolo26{model_size}-pose.pt"))
        self.model_tier = model_size

        self.head_estimator = HeadGazeEstimator(_SCRIPT_DIR)
        self.hand_estimator = HandSignalDetector(_SCRIPT_DIR)
        self.gaze = GazeTracker(self.head_estimator, batch_size=6)
        self.phone_scan = PhoneProximityScanner(batch_size=3)
        self.hand_raises = HandRaiseTracker()
        self.watch_loader = SmartwatchModelLoader(_SCRIPT_DIR)

        self.seats = SeatRegistry()
        self.paper_track = PaperOwnershipTracker()
        self.checks = ActiveChecks(phone=True, book=True, smartwatch=True, hand=True)

        self.db = Database(os.path.join(_SCRIPT_DIR, "proctor_system.db"))

        self.student_db = {}   # sid -> accumulator state (see _ensure)
        self.cooldowns = {}    # f"{sid}_{vtype}" -> epoch

        self.session_id = None
        self.session_start = None
        self.exam_duration_sec = None

        self.set_preset(DEFAULT_PRESET)
        self._warm_up()

    # ── setup ──────────────────────────────────────────────────────
    def _warm_up(self):
        dummy = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)
        try:
            self.pose_model.track(dummy, persist=True, classes=[CLS_PERSON], conf=0.35,
                                   imgsz=self.imgsz, device=self.device, half=self.use_half,
                                   tracker="botsort.yaml", verbose=False, max_det=60)
            self.det_model.track(dummy, persist=True, classes=[CLS_PHONE, CLS_BOOK], conf=0.35,
                                  imgsz=self.imgsz, device=self.device, half=self.use_half,
                                  tracker="botsort.yaml", verbose=False, max_det=30)
        except Exception:
            pass

    def set_preset(self, name):
        p = PERF_PRESETS.get(name, PERF_PRESETS[DEFAULT_PRESET])
        self.preset_name = name if name in PERF_PRESETS else DEFAULT_PRESET
        self.imgsz = p["imgsz"]
        self.conf = p["conf"]
        self.iou = p["iou"]
        self.max_det = p["max_det"]
        self.obj_skip = p["obj_skip"]
        self.min_person_h = max(CAPTURE_H // p["min_person_h_div"], 30)
        self.gaze.set_batch_size(p["face_batch"])
        self.phone_scan.set_batch_size(p["phone_scan_batch"])
        self.smartwatch_batch = max(1, p["smartwatch_batch"])
        self._fc_obj = 0
        self._have_obj_cache = False
        self._last_phones, self._last_papers = [], []

    # ── session control ─────────────────────────────────────────────
    def start_session(self, name, duration_label):
        self.session_id = self.db.start_session(name or "Web session", camera_source="browser")
        self.session_start = time.time()
        durations = {"No Limit": None, "30 min": 30, "45 min": 45, "60 min": 60,
                     "90 min": 90, "120 min": 120, "180 min": 180}
        minutes = durations.get(duration_label)
        self.exam_duration_sec = minutes * 60 if minutes else None
        self.student_db.clear()
        self.cooldowns.clear()
        return self.session_id

    def end_session(self):
        if self.session_id is not None:
            self.db.end_session(self.session_id)
        sid = self.session_id
        self.session_id = None
        self.session_start = None
        return sid

    def exam_elapsed_str(self):
        elapsed = int(time.time() - self.session_start) if self.session_start else 0
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── per-frame accumulator (ported from AIExamProctor._accumulate) ─
    def _accumulate(self, state, key, drive, cap, decay_mult=1.0):
        now = time.time()
        ts_key = f"_{key}_ts"
        last = state.get(ts_key, now)
        dt = min(max(now - last, 0.0), 0.5)
        state[ts_key] = now
        cur = state.get(key, 0.0)
        drive = float(drive)
        if drive > 0:
            state[key] = min(cur + dt * drive, cap)
        else:
            state[key] = max(cur - dt * decay_mult, 0.0)
        return state[key]

    def _ensure(self, sid):
        if sid not in self.student_db:
            self.student_db[sid] = {
                "pf": 0.0, "cf": 0.0, "mv": 0.0, "cl": 0.0, "wa": 0.0,
                "cf_warn": 0, "cf_warn_ts": None,
                "home": None, "nb_baseline": None,
                "calibrated": False, "created_at": time.time(),
                "calib_pos": [], "calib_nb": [],
            }

    def _can_fire(self, sid, vtype):
        key = f"{sid}_{vtype}"
        now = time.time()
        if key in self.cooldowns and now - self.cooldowns[key] < self.COOL_SEC:
            return False
        self.cooldowns[key] = now
        return True

    def _feed_calibration(self, sid, sbox, nearest_norm_d):
        state = self.student_db[sid]
        if state.get("calibrated"):
            return
        sx1, sy1, sx2, sy2 = sbox
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        state["calib_pos"].append((cx, cy))
        if nearest_norm_d is not None:
            state["calib_nb"].append(nearest_norm_d)
        elapsed = time.time() - state["created_at"]
        if elapsed >= self.CALIB_SEC and len(state["calib_pos"]) >= self.CALIB_MIN_SAMPLES:
            xs = [p[0] for p in state["calib_pos"]]
            ys = [p[1] for p in state["calib_pos"]]
            state["home"] = (statistics.median(xs), statistics.median(ys))
            if state["calib_nb"]:
                state["nb_baseline"] = statistics.median(state["calib_nb"])
            state["calibrated"] = True
            state["calib_pos"], state["calib_nb"] = [], []

    def _suspicion(self, sid, sf):
        st = self.student_db.get(sid, {})
        if not st.get("calibrated"):
            return 0.0, "PHONE"
        scores = {
            "PHONE":       st.get("pf", 0) / self.PHONE_SEC,
            "COPYING":     st.get("cf", 0) / self.COPY_SEC,
            "PAPER":       sf.paper_pending,
            "OUT_OF_SEAT": st.get("mv", 0) / self.SEAT_SEC,
            "COLLAB":      st.get("cl", 0) / self.COLLAB_SEC,
            "SMARTWATCH":  st.get("wa", 0) / self.WATCH_SEC,
        }
        vtype = max(scores, key=scores.get)
        return min(scores[vtype], 1.0) * 100, vtype

    # ── the six behaviour checks (ported verbatim from AIExamProctor) ─
    def _chk_phone(self, sid, sbox, sf, phones, frame):
        sx1, sy1, sx2, sy2 = sbox
        sw, sh = sx2 - sx1, sy2 - sy1
        state = self.student_db[sid]
        yolo_hit, best_pconf = False, 0.0
        for _, px1, py1, px2, py2, pconf in phones:
            pb = (px1, py1, px2, py2)
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            iou = box_iou(sbox, pb)
            near = (iou > 0.015 or (sx1 - sw * 0.35 <= pcx <= sx2 + sw * 0.35
                                     and sy1 - sh * 0.05 <= pcy <= sy2 + sh * 0.30))
            if near and pconf > 0.38:
                yolo_hit = True
                best_pconf = max(best_pconf, pconf)
        drive = 1.0 if yolo_hit else (0.4 if sf.gesture_score > 0.72 else 0.0)
        val = self._accumulate(state, "pf", drive, self.PHONE_SEC * 1.5, decay_mult=2.0)
        if val >= self.PHONE_SEC and self._can_fire(sid, "PHONE"):
            if yolo_hit:
                conf, reason = clamp(best_pconf, 0.0, 1.0), "YOLO phone detection"
            else:
                conf, reason = clamp(sf.gesture_score * 0.75, 0.0, 1.0), "wrist-at-ear gesture (no direct box match)"
            self._fire(sid, "PHONE", frame, extra={"via_pose": not yolo_hit}, confidence=conf, reason=reason)
            state["pf"] = 0

    def _chk_copying(self, sid, sf, frame):
        state = self.student_db[sid]
        drive = (1.0 if sf.turn_source == "facemesh" else 0.55) if sf.turning_active else 0.0
        val = self._accumulate(state, "cf", drive, self.COPY_SEC * 1.5)
        if val >= self.COPY_SEC and self._can_fire(sid, "COPYING"):
            reason = ("MediaPipe face-mesh (head yaw + eye gaze)" if sf.turn_source == "facemesh"
                       else "pose-keypoint estimate (face-mesh unavailable this sample)")
            conf = clamp(sf.turn_score, 0.0, 1.0)
            now = time.time()
            last_warn_at = state.get("cf_warn_ts")
            if last_warn_at is not None and (now - last_warn_at) > self.WARN_DECAY_SEC:
                state["cf_warn"] = 0
            warnings = state.get("cf_warn", 0) + 1
            state["cf_warn"], state["cf_warn_ts"] = warnings, now
            if warnings <= self.COPY_WARNINGS:
                self._fire(sid, "COPYING", frame, confidence=conf,
                           reason=f"{reason} — warning {warnings}/{self.COPY_WARNINGS}")
            else:
                state["cf_warn"] = 0
                self._fire(sid, "COPYING", frame, confidence=conf,
                           reason=f"{reason} — final warning ({self.COPY_WARNINGS} prior warnings escalated)")
            state["cf"] = 0

    def _handle_paper_handoff(self, from_sid, to_sid, ptype, box, frame, confidence=0.75):
        pst = PAPER_STYLES.get(ptype, PAPER_STYLES["TEST_PAPER"])
        for role_sid, role, partner in ((from_sid, "sent", to_sid), (to_sid, "received", from_sid)):
            if role_sid not in self.student_db:
                continue
            if self._can_fire(role_sid, "PAPER"):
                self._fire(role_sid, "PAPER", frame,
                           extra={"paper_type": ptype, "role": role, "partner_id": str(partner)},
                           confidence=confidence, reason=f"confirmed hand-off ({pst['lbl']})")

    def _chk_moved(self, sid, sbox, frame):
        state = self.student_db[sid]
        if not state.get("calibrated") or state.get("home") is None:
            return
        sx1, sy1, sx2, sy2 = sbox
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        w = max(sx2 - sx1, 1)
        hx, hy = state["home"]
        dist = math.hypot(cx - hx, cy - hy)
        thresh = max(w * 1.65, 115)
        moved = dist > thresh
        val = self._accumulate(state, "mv", moved, self.SEAT_SEC * 1.5)
        if not moved:
            state["home"] = (hx * 0.98 + cx * 0.02, hy * 0.98 + cy * 0.02)
        if val >= self.SEAT_SEC and self._can_fire(sid, "OUT_OF_SEAT"):
            conf = clamp(dist / (thresh * 2.0), 0.5, 1.0)
            self._fire(sid, "OUT_OF_SEAT", frame, confidence=conf,
                       reason=f"drifted {dist:.0f}px from calibrated seat position")
            state["mv"] = 0

    def _chk_collab(self, sid, nearest_norm_d, frame):
        state = self.student_db[sid]
        if nearest_norm_d is None or not state.get("calibrated") or state.get("nb_baseline") is None:
            return
        baseline = state["nb_baseline"]
        leaning_in = nearest_norm_d < baseline * 0.55
        val = self._accumulate(state, "cl", leaning_in, self.COLLAB_SEC * 1.5)
        if not leaning_in:
            state["nb_baseline"] = baseline * 0.98 + nearest_norm_d * 0.02
        if val >= self.COLLAB_SEC and self._can_fire(sid, "COLLAB"):
            conf = clamp(1.0 - (nearest_norm_d / max(baseline * 0.55, 0.01)), 0.4, 1.0)
            self._fire(sid, "COLLAB", frame, confidence=conf,
                       reason=f"closed to {nearest_norm_d:.2f}× normalised gap (baseline 1.0×)")
            state["cl"] = 0

    def _chk_smartwatch(self, sid, sf, frame):
        state = self.student_db[sid]
        drive = 1.0 if sf.smartwatch_hit else 0.0
        val = self._accumulate(state, "wa", drive, self.WATCH_SEC * 1.5, decay_mult=1.5)
        if val >= self.WATCH_SEC and self._can_fire(sid, "SMARTWATCH"):
            conf = clamp(sf.smartwatch_conf, 0.0, 0.55)
            self._fire(sid, "SMARTWATCH", frame, confidence=conf,
                       reason="sustained dark wrist-worn object detected (heuristic — verify manually)")
            state["wa"] = 0

    def _smartwatch_check(self, sid, box, kp, frame, checks_snap):
        """Best-effort wrist crop check -- fails soft to (False, 0.0)
        exactly like detect_wrist_object() does when no custom watch
        model is loaded (see SmartwatchModelLoader)."""
        if not checks_snap.get("smartwatch", True) or not self.watch_loader.available or kp is None:
            return False, 0.0
        try:
            l_wrist, r_wrist = kp[9], kp[10]
        except (IndexError, TypeError):
            return False, 0.0
        x1, y1, x2, y2 = box
        scale_ref = max(x2 - x1, y2 - y1, 40)
        best_hit, best_conf = False, 0.0
        for wrist in (l_wrist, r_wrist):
            if wrist[2] < 0.35:
                continue
            hit, conf = detect_wrist_object(frame, (wrist[0], wrist[1]), scale_ref,
                                             watch_model=self.watch_loader.model,
                                             class_id=self.watch_loader.class_id)
            if hit and conf > best_conf:
                best_hit, best_conf = True, conf
        return best_hit, best_conf

    # ── evidence + persistence (ported from AIExamProctor._fire) ──────
    def _fire(self, sid, vtype, frame, extra=None, confidence=1.0, reason=""):
        ts = datetime.now()
        key = ts.strftime("%Y%m%d_%H%M%S_%f")
        exam_elapsed = self.exam_elapsed_str()
        fname = f"ID{sid}_{vtype}_{key}.jpg"
        ev_path = os.path.join(EVIDENCE_DIR, fname)
        try:
            ev_frame = frame.copy()
            st = VSTYLES.get(vtype, VSTYLES["PHONE"])
            cv2.putText(ev_frame, f"{st['icon']} {st['lbl']} · S{sid} · {confidence*100:.0f}%",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if reason:
                cv2.putText(ev_frame, reason[:90], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imwrite(ev_path, ev_frame)
        except Exception:
            ev_path = ""

        ok, row_id = self.db.add_violation(
            student_id=sid, violation_type=vtype, evidence_path=ev_path,
            session_id=self.session_id, datetime_str=ts.strftime("%Y-%m-%d %H:%M:%S"),
            exam_elapsed=exam_elapsed, confidence=confidence, reason=reason, extra=extra,
        )
        record = {
            "id": row_id if ok else None, "student_id": str(sid), "violation_type": vtype,
            "evidence_path": ev_path, "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "exam_elapsed": exam_elapsed, "confidence": round(float(confidence), 2), "reason": reason,
        }
        if extra:
            record.update(extra)
        self._new_violations.append(record)

    # ── main per-frame entry point (ported from InferenceWorker._process
    #     + AIExamProctor._on_frame, merged into one synchronous call) ──
    def process_frame(self, frame):
        now = time.time()
        self._new_violations = []

        pose_res = self.pose_model.track(
            frame, persist=True, classes=[CLS_PERSON], conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.device, half=self.use_half,
            tracker="botsort.yaml", verbose=False, max_det=self.max_det,
        )

        raw = []
        if pose_res and pose_res[0].boxes is not None and pose_res[0].boxes.id is not None:
            boxes = pose_res[0].boxes
            kpt_data = (pose_res[0].keypoints.data.cpu().numpy()
                        if pose_res[0].keypoints is not None else None)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                bw, bh = x2 - x1, y2 - y1
                if confs[i] < self.conf or bh < self.min_person_h or bw < 18:
                    continue
                box = (int(x1), int(y1), int(x2), int(y2))
                kp = kpt_data[i] if kpt_data is not None else None
                raw.append((int(ids[i]), box, kp, float(confs[i])))

        seat_map, rebound = self.seats.resolve_all([(t[0], t[1]) for t in raw], now)
        seat_boxes, seat_kp, seat_conf = {}, {}, {}
        for tid, box, kp, conf in raw:
            sid = seat_map.get(tid)
            if sid is None:
                continue
            seat_boxes[sid] = box
            seat_kp[sid] = kp
            seat_conf[sid] = conf

        checks = self.checks.snapshot()
        phone_on = checks.get("phone", True)
        classes_needed = [CLS_BOOK] + ([CLS_PHONE] if phone_on else [])

        self._fc_obj = getattr(self, "_fc_obj", 0) + 1
        run_obj = (self._fc_obj % max(self.obj_skip, 1) == 0) or not self._have_obj_cache
        phones, papers = self._last_phones, self._last_papers
        if run_obj:
            phones, papers = [], []
            obj_res = self.det_model.track(
                frame, persist=True, classes=classes_needed, conf=self.conf, iou=self.iou,
                imgsz=self.imgsz, device=self.device, half=self.use_half,
                tracker="botsort.yaml", verbose=False, max_det=30,
            )
            if obj_res and obj_res[0].boxes is not None:
                b = obj_res[0].boxes
                xyxy = b.xyxy.cpu().numpy()
                confs = b.conf.cpu().numpy()
                clss = b.cls.cpu().numpy()
                ids = b.id.cpu().numpy() if b.id is not None else np.full(len(xyxy), -1)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])
                    tid = int(ids[i])
                    if int(clss[i]) == CLS_PHONE:
                        phones.append((tid, x1, y1, x2, y2, float(confs[i])))
                    elif int(clss[i]) == CLS_BOOK:
                        ptype = classify_paper(frame, (x1, y1, x2, y2))
                        papers.append((tid, x1, y1, x2, y2, ptype, float(confs[i])))
            self._last_phones, self._last_papers = phones, papers
            self._have_obj_cache = True

        # phone-proximity close-up scan + paper hand-off tracking
        checks_snap0 = self.checks.snapshot()
        if checks_snap0.get("phone", True):
            self.phone_scan.scan_batch(
                self.det_model, frame, seat_boxes, self.conf, self.iou,
                self.device, self.use_half, CLS_PHONE, now,
            )
            extra_phones = self.phone_scan.current_detections(now)
            if extra_phones:
                phones = merge_phone_detections(phones, extra_phones)

        handoffs = []
        for tid, x1, y1, x2, y2, ptype, pconf in papers:
            box = (x1, y1, x2, y2)
            frm, to, hconf = self.paper_track.update(tid, box, seat_boxes, now, det_conf=pconf)
            if frm is not None and to is not None:
                handoffs.append((frm, to, ptype, box, hconf))
        self.paper_track.forget_stale(set(seat_boxes.keys()))

        # gaze / head-turn
        seats_out = []
        nearest_norm = {}
        for sid, sbox in seat_boxes.items():
            sx1, sy1, sx2, sy2 = sbox
            scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            sh = max(sy2 - sy1, 1)
            best_d = None
            for oid, obox in seat_boxes.items():
                if oid == sid:
                    continue
                ox1, oy1, ox2, oy2 = obox
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                oh = max(oy2 - oy1, 1)
                d = math.hypot(scx - ocx, scy - ocy) / ((sh + oh) / 2.0)
                if best_d is None or d < best_d:
                    best_d = d
            nearest_norm[sid] = best_d

        seat_list = list(seat_boxes.items())
        self.gaze.sample_batch(frame, seat_list, now)

        for sid, box in seat_boxes.items():
            self._ensure(sid)
            self._feed_calibration(sid, box, nearest_norm.get(sid))

        for sid, box in seat_boxes.items():
            kp = seat_kp.get(sid)
            turning_active, turn_score, turn_source = self.gaze.score(sid, kp, now)
            gesture_score = score_phone_gesture(kp) if kp is not None else 0.0
            pending = self.paper_track.pending_ratio(sid)
            sw_hit, sw_conf = self._smartwatch_check(sid, box, kp, frame, checks_snap0)

            sf = SeatFrame(
                seat_id=sid, box=box, conf=seat_conf.get(sid, 0.0), rebound=sid in rebound,
                turning_active=turning_active, turn_score=turn_score, turn_source=turn_source,
                gesture_score=gesture_score, paper_pending=pending,
                smartwatch_hit=sw_hit, smartwatch_conf=sw_conf,
            )
            pct, vtype = self._suspicion(sid, sf)
            calibrated = self.student_db.get(sid, {}).get("calibrated")
            seats_out.append({
                "seat_id": sid, "box": list(box), "pct": round(pct, 1), "vtype": vtype,
                "calibrated": bool(calibrated),
            })

            checks_snap = self.checks.snapshot()
            if checks_snap.get("phone", True):
                self._chk_phone(sid, box, sf, phones, frame)
            self._chk_copying(sid, sf, frame)
            self._chk_moved(sid, box, frame)
            self._chk_collab(sid, nearest_norm.get(sid), frame)
            if checks_snap.get("smartwatch", True):
                self._chk_smartwatch(sid, sf, frame)

        for from_sid, to_sid, ptype, pbox, hconf in handoffs:
            self._handle_paper_handoff(from_sid, to_sid, ptype, pbox, frame, hconf)

        return {
            "seats": seats_out,
            "phones": [[p[1], p[2], p[3], p[4], round(p[5], 2)] for p in phones],
            "papers": [[p[1], p[2], p[3], p[4], p[5]] for p in papers],
            "student_count": len(seats_out),
            "violations": self._new_violations,
            "exam_elapsed": self.exam_elapsed_str(),
        }


# ─────────────────────────── Flask app ─────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
engine = None  # lazily created on first request so `flask run` boots fast


def get_engine():
    global engine
    if engine is None:
        engine = ProctorEngine()
    return engine


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/evidence.html")
def evidence_page():
    return send_from_directory("static", "evidence.html")


@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/api/status")
def api_status():
    eng = get_engine()
    return jsonify({
        "device": eng.dev_label, "model_tier": eng.model_tier,
        "preset": eng.preset_name, "presets": list(PERF_PRESETS.keys()),
        "checks": eng.checks.snapshot(),
        "session_active": eng.session_id is not None,
        "exam_elapsed": eng.exam_elapsed_str() if eng.session_id else None,
    })


@app.route("/api/session/start", methods=["POST"])
def api_session_start():
    eng = get_engine()
    data = request.get_json(force=True, silent=True) or {}
    sid = eng.start_session(data.get("name", ""), data.get("duration", "No Limit"))
    return jsonify({"session_id": sid})


@app.route("/api/session/end", methods=["POST"])
def api_session_end():
    eng = get_engine()
    eng.end_session()
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    eng = get_engine()
    data = request.get_json(force=True, silent=True) or {}
    if "preset" in data:
        eng.set_preset(data["preset"])
    if "checks" in data:
        eng.checks.update(data["checks"])
    return jsonify({"ok": True})


@app.route("/api/frame", methods=["POST"])
def api_frame():
    eng = get_engine()
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame uploaded"}), 400
    buf = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400
    if frame.shape[1] != CAPTURE_W or frame.shape[0] != CAPTURE_H:
        frame = cv2.resize(frame, (CAPTURE_W, CAPTURE_H))
    try:
        result = eng.process_frame(frame)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


@app.route("/api/violations")
def api_violations():
    eng = get_engine()
    student_id = request.args.get("student_id")
    session_id = request.args.get("session_id", type=int)
    return jsonify(eng.db.get_violations(student_id=student_id, session_id=session_id))


@app.route("/api/violations/<int:violation_id>", methods=["DELETE"])
def api_violation_delete(violation_id):
    eng = get_engine()
    ok = eng.db.delete_violation(violation_id)
    return jsonify({"ok": ok})


@app.route("/api/violations/clear", methods=["POST"])
def api_violations_clear():
    eng = get_engine()
    eng.db.clear_violations()
    return jsonify({"ok": True})


@app.route("/api/violation_stats")
def api_violation_stats():
    eng = get_engine()
    return jsonify(eng.db.get_violation_stats())


@app.route("/api/students", methods=["GET", "POST"])
def api_students():
    eng = get_engine()
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        ok, msg = eng.db.add_student(data.get("student_name", ""), data.get("student_id", ""),
                                      data.get("class_name", ""))
        return jsonify({"ok": ok, "message": msg})
    return jsonify(eng.db.get_students())


if __name__ == "__main__":
    print("Loading models... this can take a moment the first time.")
    get_engine()
    print("EYE web is ready → open http://localhost:5000 in a browser on this machine")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
