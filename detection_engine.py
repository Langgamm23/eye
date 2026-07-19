"""
detection_engine.py
────────────────────
Detection / identity engine for EYE. Split out of eye.py so the actual
computer-vision and tracking logic — the part that decides whether
something is a violation — can be read, tested, and tuned on its own,
independent of the PyQt6 application shell. Nothing in this file
imports PyQt6; it only needs opencv, numpy, and (optionally) mediapipe.

Contents
────────
  geometry helpers    box_iou, box_center, box_wh, clamp
  ScoreGate           EMA smoothing + hysteresis for any 0..1 signal,
                       used by every behaviour check so a value sitting
                       right on a threshold doesn't flicker the result
                       every other frame.
  SeatRegistry        Turns ephemeral BoT-SORT track IDs into stable
                       per-seat identities that survive brief tracking
                       drop-outs (occlusion, re-entry, a tracker ID
                       switch) instead of silently starting a fresh,
                       empty violation history for the same student.
  PaperOwnershipTracker
                       Turns "a paper-like object is currently near two
                       students" into "a paper-like object just changed
                       hands" — a real hand-off, confirmed over time,
                       instead of a static-position guess.
  check_paper_passing  v5.2: a second, more direct paper-hand-off
                       signal -- fires only when two adjacent students'
                       hands are genuinely close together (scipy
                       euclidean distance on their YOLO-pose wrist
                       keypoints) AND a YOLO paper box overlaps that
                       exact hand-to-hand zone. Complements
                       PaperOwnershipTracker rather than replacing it.
  score_phone_gesture  Tightened wrist-near-ear heuristic, used only as
                       a secondary, slower-accumulating corroborator to
                       the YOLO phone detector (not a standalone trigger).
  fallback_turn_score  Improved pose-keypoints-only head-turn estimate,
                       used automatically when MediaPipe is unavailable
                       or hasn't sampled this student yet.
  HeadGazeEstimator    MediaPipe Face-Landmarker wrapper: crops a
                       student's head from the full-resolution frame,
                       upscales it (this is what keeps accuracy up for
                       students far from the camera), and runs OpenCV
                       solvePnP against a generic 3D face model to get
                       a REAL yaw/pitch/roll in degrees (v5.2 -- see
                       _yaw_pitch_roll_solvepnp), not just a ratio.
                       Fails soft: if mediapipe isn't installed, or its
                       model can't be downloaded (school firewall,
                       offline machine, etc.), `.available` is simply
                       False and every caller already has a working
                       fallback path.
  HeadTurnTracker      v5.2: confirms a head turn only once the real
                       yaw angle has stayed past threshold for
                       hold_seconds of WALL-CLOCK time (time.time(),
                       not a frame count) -- a quick glance that
                       crosses the angle for one noisy frame doesn't
                       fire; a sustained 1-2 second turn does. FPS-safe
                       by construction, which matters specifically
                       because this app can run slow on weak hardware.
  GazeTracker          Ties HeadGazeEstimator's per-student round-robin
                       sampling + caching + HeadTurnTracker's hold-time
                       buffer together into one call per student per
                       frame.
  detect_wrist_object  v5.2: classifies a wrist crop with a trained
                       YOLO model (your custom smartwatch-class model)
                       instead of HSV colour masking -- the old colour
                       mask had no way to tell a dark sleeve cuff or a
                       shadow apart from an actual watch; a trained
                       detector has actually seen labelled examples of
                       what a watch looks like. Fails soft to "not
                       detected" if no watch model is supplied.
  classify_paper       Classical-CV sub-typing of a detected paper-like
                       object (yellow pad / bubble sheet / test paper).
  list_available_cameras
                       Probes camera indices so the app can offer a
                       real "here's what's plugged in" list instead of
                       a hard-coded two-camera dropdown — this is what
                       lets a USB webcam just show up as another option.
"""

import os
import math
import time
import shutil
import threading
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import distance as _scipy_distance


# ─────────────────────────── worker → UI data contract ───────────────

@dataclass
class SeatFrame:
    """Everything the UI thread needs for one tracked student, one frame."""
    seat_id: int
    box: tuple              # (x1, y1, x2, y2) ints, in full-frame coords
    conf: float
    rebound: bool            # True the one frame a seat was just re-bound
    turning_active: bool     # hysteresis-gated "looking away" state
    turn_score: float        # 0..1 smoothed score behind that state
    turn_source: str         # "facemesh" | "pose"
    gesture_score: float     # 0..1 phone-near-ear corroborator
    paper_pending: float     # 0..1, for the live risk badge only
    smartwatch_hit: bool = False        # experimental -- see detect_wrist_object
    smartwatch_conf: float = 0.0


@dataclass
class FrameAnalysis:
    """One frame's worth of perception, handed from InferenceWorker to the UI thread."""
    seats: list = field(default_factory=list)      # List[SeatFrame]
    phones: list = field(default_factory=list)      # List[(tid,x1,y1,x2,y2,conf)]
    papers: list = field(default_factory=list)      # List[(tid,x1,y1,x2,y2,ptype,conf)] -- ptype may be "NOTEBOOK"
    handoffs: list = field(default_factory=list)    # List[(from_seat,to_seat,ptype,box,confidence)]
    hand_signals: list = field(default_factory=list)  # List[(seat_id, finger_count)] -- CONFIRMED events only
    hand_tracking: list = field(default_factory=list)  # List[(seat_id, [(x,y) x21])] -- for live skeleton drawing
    face_mesh_status: str = ""
    hand_mesh_status: str = ""


# ─────────────────────────── geometry helpers ───────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box_iou(a, b):
    """Intersection-over-union for two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_wh(b):
    return (b[2] - b[0], b[3] - b[1])


# ─────────────────────────── camera enumeration ─────────────────────

def list_available_cameras(max_probe=6):
    """
    Actually probes camera indices 0..max_probe-1 (instead of assuming
    a fixed 'built-in + external' pair) so a USB webcam -- which
    normally just takes the next free index -- shows up automatically
    as its own selectable entry. Each probe is a real open/read/close,
    so this takes a moment; call it once at startup and again only if
    the user asks to refresh (e.g. after plugging something in).
    Returns a list of dicts: [{"index": 0, "label": "Camera 0 (1280x720)"}, ...]
    Never raises -- a platform/driver hiccup on one index just means
    that index is skipped, not that the whole scan fails.
    """
    found = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for idx in range(max_probe):
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if ok and frame is not None:
                label = f"Camera {idx}" + (f"  ({w}x{h})" if w and h else "")
                found.append({"index": idx, "label": label})
        except Exception:
            continue
    return found


# ─────────────────────────── thread-safe teacher toggles ─────────────

class ActiveChecks:
    """
    Thread-safe shared toggle state: the UI thread WRITES it the
    instant a teacher clicks a checkbox, InferenceWorker READS it every
    frame. Same lock-protected-shared-state pattern as FrameMailbox
    (see eye.py) rather than a Qt cross-thread signal, since this is
    plain state a worker polls each frame, not an event -- a lock is
    simpler to reason about here and just as thread-safe: a checkbox
    toggle takes effect on literally the next frame the worker
    processes, no restart needed.

    Keys are short, lowercase, and match what the UI exposes directly:
    "phone", "book" (notebook/book-like objects), "smartwatch", "hand"
    (hand-raise/finger-count signalling). Unlisted/unknown keys default
    to True (fail open -- an unrecognised key should never silently
    disable a check that was never meant to be toggled).
    """
    def __init__(self, **defaults):
        self._lock = threading.Lock()
        self._state = dict(defaults)

    def set(self, key, value):
        with self._lock:
            self._state[key] = bool(value)

    def update(self, mapping):
        with self._lock:
            self._state.update({k: bool(v) for k, v in mapping.items()})

    def is_on(self, key, default=True):
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self):
        with self._lock:
            return dict(self._state)


# ─────────────────────────── hysteresis gate ─────────────────────────

class ScoreGate:
    """
    Turns a noisy per-frame 0..1 score into a stable boolean using EMA
    smoothing plus hysteresis (a higher threshold to turn on than to
    turn off). Without this, a raw score bouncing right around a
    single cutoff flips the result every other frame, which is exactly
    the kind of noise that used to make a hard single-threshold check
    unreliable. `enter` and `exit_` should satisfy enter > exit_.
    """
    __slots__ = ("enter", "exit_", "alpha", "smoothed", "active")

    def __init__(self, enter, exit_, alpha=0.35):
        self.enter = enter
        self.exit_ = exit_
        self.alpha = alpha
        self.smoothed = None
        self.active = False

    def update(self, raw_score):
        raw_score = clamp(float(raw_score), 0.0, 1.0)
        self.smoothed = raw_score if self.smoothed is None else \
            self.smoothed + self.alpha * (raw_score - self.smoothed)
        if self.active:
            if self.smoothed < self.exit_:
                self.active = False
        else:
            if self.smoothed > self.enter:
                self.active = True
        return self.active, self.smoothed

    def reset(self):
        self.smoothed = None
        self.active = False


# ─────────────────────────── seat identity registry ──────────────────

class SeatRegistry:
    """
    BoT-SORT (or any tracker) hands out track IDs that can churn --
    a student gets briefly occluded by someone walking past, or the
    tracker just hiccups, and the same physical student comes back a
    frame later under a brand-new ID. Every accumulated violation
    counter, cooldown, and seat baseline was keyed off that ID, so an
    ID switch used to silently reset all of it and could even split
    one student's history across multiple "student IDs" in the log.

    SeatRegistry fixes this the same way a substitute teacher would:
    by seat, not by face. Each frame it's handed the raw (track_id,
    box) detections; if a brand-new track_id appears close to where a
    seat was last seen within REBIND_WINDOW seconds, it's treated as
    the same seat continuing, not a new one. Genuinely new positions
    (a seat that's been empty for a while, or a spot nobody's occupied
    before) still get a fresh seat_id.

    This only helps with tracker churn for a student who is roughly
    where they were -- a student who actually leaves and a different
    student who then sits in a totally different spot still correctly
    get different seat_ids.

    Two things changed from the original version, both aimed at the
    same complaint -- a student's number silently changing mid-exam:
      - REBIND_WINDOW went from 4s to 45s. 4 seconds is shorter than a
        lot of completely ordinary occlusion: a teacher walking the
        width of the room, another student standing up in front of
        the camera to hand in a paper, or the object detector just
        missing a couple frames in a row. Exam-hall seating doesn't
        rearrange itself mid-session, so there's little downside to
        waiting much longer before a seat is considered actually
        vacated and its number retired.
      - Matching is now a single global nearest-first assignment
        instead of resolving each unresolved detection in whatever
        order the tracker happened to list them. The old order-
        dependent version could let detection A grab a seat that was
        actually a closer, better match for detection B (resolved
        later in the same loop) -- most likely to happen exactly when
        several students go untracked at once, which is also the
        moment seat continuity matters most. A height-compatibility
        check was also added as a second gate alongside distance, so a
        much longer window doesn't make it easier to accidentally bind
        onto an unrelated person (e.g. the teacher) who merely passed
        near a vacated seat -- height is used rather than width for
        the same reason _crop_head prefers it elsewhere: a seated
        student's width swings a lot with arm/pose, height stays far
        more stable.
    """
    REBIND_WINDOW = 45.0   # seconds a vacated seat stays "claimable"
    MIN_RADIUS    = 70     # px floor for the rebind search radius
    HEIGHT_RATIO_MIN = 0.5   # candidate box height vs. the seat's last known
    HEIGHT_RATIO_MAX = 1.8   # height must fall in this range to be considered

    def __init__(self):
        self._seats = {}          # seat_id -> dict(pos, w, h, last_seen, track_id)
        self._track_to_seat = {}  # track_id -> seat_id
        self._next_id = 1

    def resolve_all(self, detections, now):
        """
        detections: iterable of (track_id, box) for this frame's raw
        person detections (box = (x1,y1,x2,y2)).
        Returns {track_id: seat_id} for this frame. Also returns the
        set of seat_ids that were just re-bound to a new track_id this
        call (useful for a one-line debug log), as the second element
        of a tuple: (mapping, rebound_seat_ids).
        """
        result = {}
        rebound = set()
        claimed = set()
        unresolved = []

        for track_id, box in detections:
            seat_id = self._track_to_seat.get(track_id)
            if seat_id is not None and seat_id in self._seats:
                cx, cy = box_center(box); w, h = box_wh(box)
                self._seats[seat_id].update(pos=(cx, cy), w=w, h=h,
                                             last_seen=now, track_id=track_id)
                result[track_id] = seat_id
                claimed.add(seat_id)
            else:
                unresolved.append((track_id, box))

        # Build every plausible (track, seat) rebind candidate up
        # front, then assign nearest-first globally -- see the class
        # docstring for why this replaced the old per-detection,
        # order-dependent loop.
        candidates = []
        for track_id, box in unresolved:
            cx, cy = box_center(box); w, h = box_wh(box)
            radius = max(w * 1.3, self.MIN_RADIUS)
            for sid, s in self._seats.items():
                if sid in claimed:
                    continue
                gap = now - s["last_seen"]
                if not (0 < gap <= self.REBIND_WINDOW):
                    continue
                ref_h = s.get("h") or h
                if ref_h > 0 and not (self.HEIGHT_RATIO_MIN <= h / ref_h <= self.HEIGHT_RATIO_MAX):
                    continue
                d = math.hypot(cx - s["pos"][0], cy - s["pos"][1])
                if d < radius:
                    candidates.append((d, track_id, sid, cx, cy, w, h))

        candidates.sort(key=lambda c: c[0])
        used_tracks = set()
        for d, track_id, sid, cx, cy, w, h in candidates:
            if track_id in used_tracks or sid in claimed:
                continue
            old_track = self._seats[sid].get("track_id")
            if old_track is not None:
                self._track_to_seat.pop(old_track, None)
            self._track_to_seat[track_id] = sid
            self._seats[sid].update(pos=(cx, cy), w=w, h=h,
                                     last_seen=now, track_id=track_id)
            claimed.add(sid)
            used_tracks.add(track_id)
            rebound.add(sid)
            result[track_id] = sid

        for track_id, box in unresolved:
            if track_id in used_tracks:
                continue
            cx, cy = box_center(box); w, h = box_wh(box)
            seat_id = self._next_id
            self._next_id += 1
            self._seats[seat_id] = dict(pos=(cx, cy), w=w, h=h,
                                         last_seen=now, track_id=track_id)
            self._track_to_seat[track_id] = seat_id
            claimed.add(seat_id)
            result[track_id] = seat_id

        return result, rebound

    def is_known(self, seat_id):
        return seat_id in self._seats


# ─────────────────────────── paper hand-off tracking ──────────────────

class PaperOwnershipTracker:
    """
    The old check fired whenever a paper-like object simply sat between
    two adjacent students -- which, in a real classroom with students
    seated close together, describes an enormous number of completely
    innocent moments (everyone's own paper is "between" them and their
    neighbour). This tracks who a given tracked paper's nearest student
    is over time and only fires on a *confirmed change* of nearest
    student -- i.e. an actual hand-off -- which is what a paper-passing
    violation should really mean.

    Two refinements on top of that baseline:
      - Distance is perspective-normalised (divided by each candidate
        student's own seat-box height) instead of raw pixels, same
        reasoning as the COLLAB check -- otherwise whichever student
        happens to be closest to the camera (biggest box) unfairly
        wins "nearest" regardless of who the paper is actually next
        to.
      - A switch requires a clear margin over the current owner's own
        (also normalised) distance before the pending clock even
        starts -- a paper sitting near the midpoint of two adjacent
        desks will otherwise have its "nearest" flicker between them
        on ordinary tracking jitter alone.
    """
    CONFIRM_SEC    = 0.35   # candidate new owner must hold this long to confirm
    MAX_LINK_RATIO = 0.9    # normalised distance (÷ seat height); farther than
                             # this from every student isn't linked to anyone
    SWITCH_MARGIN  = 0.75   # a new candidate must be this much closer than the
                             # current owner (ratio) before it's even considered

    def __init__(self):
        self._papers = {}   # paper_track_id -> state dict

    @staticmethod
    def _nearest(paper_box, seat_boxes):
        """Perspective-normalised nearest seat: raw center-to-center
        distance divided by that seat's own box height, so students
        closer to the camera don't automatically win. Returns
        (seat_id, normalised_dist) or (None, inf) if seat_boxes is
        empty."""
        pcx, pcy = box_center(paper_box)
        best_id, best_d = None, float("inf")
        for sid, box in seat_boxes.items():
            scx, scy = box_center(box)
            sh = max(box[3] - box[1], 1)
            d = math.hypot(pcx - scx, pcy - scy) / sh
            if d < best_d:
                best_d, best_id = d, sid
        return best_id, best_d

    def update(self, paper_track_id, paper_box, seat_boxes, now, det_conf=1.0):
        """
        seat_boxes: {seat_id: (x1,y1,x2,y2)} for this frame's students.
        det_conf: the object detector's own confidence (0..1) for this
        paper box this frame -- folded into the confidence returned on
        a confirmed hand-off, so a shaky low-confidence detection
        doesn't get reported with the same certainty as a clean one.

        Returns (from_seat, to_seat, confidence) the instant a hand-off
        is confirmed, else (None, None, 0.0). Tracks its own elapsed
        time per paper internally (clamped, same reasoning as the
        accumulator elsewhere: a stall shouldn't register as one huge
        jump) rather than trusting the caller to hand in a correct dt
        every time.
        """
        nearest_id, nearest_d = self._nearest(paper_box, seat_boxes)
        if nearest_d > self.MAX_LINK_RATIO:
            nearest_id = None

        st = self._papers.get(paper_track_id)
        if st is None:
            self._papers[paper_track_id] = {
                "owner": nearest_id, "pending": None, "pending_t": 0.0, "last_ts": now,
            }
            return None, None, 0.0

        dt = clamp(now - st["last_ts"], 0.0, 0.5)
        st["last_ts"] = now

        if nearest_id == st["owner"]:
            st["pending"] = None
            st["pending_t"] = 0.0
            return None, None, 0.0

        # Distance from the paper to its CURRENT owner this frame (if
        # that student is still visible) -- the margin check below
        # needs this to decide whether the new candidate is a real
        # switch or just jitter near the boundary.
        owner_box = seat_boxes.get(st["owner"])
        owner_d = None
        if owner_box is not None:
            _, owner_d = self._nearest(paper_box, {st["owner"]: owner_box})

        if owner_d is not None and nearest_id is not None and nearest_d > owner_d * self.SWITCH_MARGIN:
            # Not clearly closer to the candidate than to the current
            # owner -- treat as noise, don't even start the clock.
            st["pending"] = None
            st["pending_t"] = 0.0
            return None, None, 0.0

        if nearest_id == st["pending"]:
            st["pending_t"] += dt
        else:
            st["pending"] = nearest_id
            st["pending_t"] = dt

        if st["pending_t"] >= self.CONFIRM_SEC:
            held = st["pending_t"]
            old_owner = st["owner"]
            st["owner"] = nearest_id
            st["pending"] = None
            st["pending_t"] = 0.0
            if old_owner is not None and nearest_id is not None and old_owner != nearest_id:
                # Confidence blends three signals: how decisively the
                # paper left the old owner's side (bigger margin =
                # more confident), how long it held with the new
                # owner beyond the bare minimum confirm window, and
                # the detector's own confidence that this box is a
                # paper at all.
                margin_conf = clamp(1.0 - (nearest_d / max(owner_d, 0.01)), 0.0, 1.0) if owner_d else 0.5
                hold_conf   = clamp((held - self.CONFIRM_SEC) / self.CONFIRM_SEC, 0.0, 1.0)
                confidence  = clamp(0.4 + 0.3 * margin_conf + 0.15 * hold_conf + 0.15 * det_conf, 0.0, 1.0)
                return old_owner, nearest_id, confidence
        return None, None, 0.0

    def forget_stale(self, active_track_ids):
        for tid in list(self._papers):
            if tid not in active_track_ids:
                del self._papers[tid]

    def pending_ratio(self, seat_id):
        """
        0..1 -- how far a hand-off *to* this seat has progressed toward
        being confirmed, for the live on-box risk badge only. Purely
        cosmetic/informational; firing itself is decided entirely by
        `update()` above.
        """
        best = 0.0
        for st in self._papers.values():
            if st["pending"] == seat_id and st["owner"] != seat_id:
                best = max(best, clamp(st["pending_t"] / self.CONFIRM_SEC, 0.0, 1.0))
        return best


# ─────────────────────────── phone gesture scoring ────────────────────

def score_phone_gesture(kp):
    """
    Wrist held close to an EAR specifically, roughly jaw-level or
    above -- a much tighter zone than the original 'wrist anywhere
    near the head' check, which routinely fired on ordinary thinking
    gestures (chin on fist, touching hair, adjusting glasses). Returns
    a continuous 0..1 score rather than a bare bool, so the caller can
    treat this as slower-accumulating corroborating evidence rather
    than full-strength proof on its own -- it's meant to catch phones
    a YOLO box detector misses at desk distance, not to fire alone.
    kp: one person's (17,3) COCO keypoints array [x,y,confidence].
    """
    if kp is None:
        return 0.0
    try:
        nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = kp[0:7]
        l_wrist, r_wrist = kp[9], kp[10]
    except (IndexError, TypeError):
        return 0.0

    ears = [e for e in (l_ear, r_ear) if e[2] > 0.30]
    if not ears or nose[2] < 0.35:
        return 0.0
    sh_w = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
    if sh_w < 20:
        return 0.0
    zone = sh_w * 0.45   # a tight, ear-sized radius, not "anywhere near the head"

    best = 0.0
    for wrist in (l_wrist, r_wrist):
        if wrist[2] < 0.35:
            continue
        for ear in ears:
            d = math.hypot(wrist[0] - ear[0], wrist[1] - ear[1])
            if d < zone:
                best = max(best, clamp(1.0 - d / zone, 0.0, 1.0))
    return best


# ─────────────────────────── wrist object heuristic (smartwatch) ──────

def _crop_wrist_band(frame, wrist_xy, scale_ref, target=96):
    """Small square crop tightly around a wrist point, upscaled."""
    h_frame, w_frame = frame.shape[:2]
    wx, wy = wrist_xy
    half = clamp(scale_ref * 0.35, 14, 90)
    x1 = int(clamp(wx - half, 0, w_frame)); x2 = int(clamp(wx + half, 0, w_frame))
    y1 = int(clamp(wy - half, 0, h_frame)); y2 = int(clamp(wy + half, 0, h_frame))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    square = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                 cv2.BORDER_REPLICATE)
    if side != target:
        square = cv2.resize(square, (target, target), interpolation=cv2.INTER_LINEAR)
    return square


# v5.2 default settings for the YOLO-based smartwatch check below --
# WATCH_CLASS_ID must match whatever index "smartwatch" was given in
# YOUR custom model's data.yaml (0 is just this file's default; check
# your own training config). WATCH_CONF_THRESHOLD/WATCH_INFER_SIZE are
# safe starting points to tune against your own model's validation
# results.
WATCH_CLASS_ID       = 0
WATCH_CONF_THRESHOLD = 0.45
WATCH_INFER_SIZE      = 224


class SmartwatchModelLoader:
    """
    Loads the custom-trained YOLO model detect_wrist_object() needs --
    this check has no data of its own without one, since (as that
    function's docstring says) COCO ships no watch class at all. This
    used to simply never get wired up anywhere in eye.py, so the
    smartwatch check silently did nothing regardless of the "Detect
    Smartwatch" toggle.

    Fails soft exactly like HeadGazeEstimator/HandSignalDetector
    elsewhere in this file: no model file found -> .available is
    False, .model is None, and detect_wrist_object() already treats a
    None model as "check disabled" rather than raising. There's
    nothing to auto-download here the way the MediaPipe models are --
    those are public, unauthenticated files on Google's storage,
    while a custom smartwatch weight has to come from somewhere that
    was actually trained on watches (e.g. Roboflow Universe), which
    means a one-time manual download tied to an account rather than a
    silent first-run fetch. See the class_id note below: the index
    the model calls "smart watch" is set by whoever trained it, not by
    this file, so it must be read from that model's own data.yaml
    rather than assumed.

    To use a real model:
      1. Get a trained weights file with a smartwatch/watch class --
         e.g. Roboflow Universe's "cheating detection" project
         (kiros-workspace-ywsvn/cheating-detection-ghpja, purpose-built
         for exam monitoring: book/phone/laptop/earphone/headphones/
         smart watch all in one) or a watch-only project such as
         vincent-y20l1/smart-watch. Downloading a Roboflow-hosted
         weights file needs a free Roboflow account and API key --
         unlike this file's other model downloads, it isn't a plain
         unauthenticated URL, so that step has to happen once, by hand,
         outside this app.
      2. Save the .pt file to models/smartwatch.pt next to eye.py (or
         point EYE_WATCH_MODEL_PATH at wherever you put it).
      3. Open the model's data.yaml (Roboflow includes this with every
         export) and find the integer index of the smart-watch/watch
         class -- set that as class_id here (or WATCH_CLASS_ID above)
         to match. Getting this wrong doesn't crash anything; it just
         means the check silently watches for the wrong class.
    """

    def __init__(self, script_dir, class_id=WATCH_CLASS_ID):
        self.available = False
        self.status = "not initialised"
        self.model = None
        self.class_id = class_id
        self._try_init(script_dir)

    def _try_init(self, script_dir):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.status = f"ultralytics not installed ({e}) — smartwatch check disabled"
            return

        model_path = os.environ.get("EYE_WATCH_MODEL_PATH")
        if not model_path:
            model_path = os.path.join(script_dir, "models", "smartwatch.pt")
        if not os.path.exists(model_path):
            self.status = (f"no smartwatch model at {model_path} — smartwatch check "
                            f"disabled (see SmartwatchModelLoader's docstring to add one)")
            return

        try:
            self.model = YOLO(model_path)
            self.available = True
            self.status = f"smartwatch model loaded from {model_path}"
        except Exception as e:
            self.status = f"failed to load smartwatch model ({e}) — smartwatch check disabled"
            self.model = None


def detect_wrist_object(frame, wrist_xy, scale_ref, watch_model=None,
                         class_id=WATCH_CLASS_ID, conf_threshold=WATCH_CONF_THRESHOLD,
                         infer_size=WATCH_INFER_SIZE):
    """
    v5.2 -- replaces the old HSV colour-mask heuristic. A colour mask
    only ever asked "is a good chunk of this patch dark?", which a
    shadow, a dark sleeve cuff, or a dark pen near the wrist all
    satisfy exactly as well as a real watch does -- there's no concept
    of "watch-shaped" in a colour threshold. This crops the same wrist
    region (via _crop_wrist_band, unchanged) but classifies it with a
    trained YOLO model instead: `watch_model` should be your own
    custom-trained ultralytics.YOLO instance that has seen labelled
    smartwatch examples, since COCO (what the base det/pose models in
    this app ship with) has no watch class at all.

    Fails soft exactly like every other optional dependency in this
    file: pass watch_model=None (the default) to disable this check
    entirely -- it returns "not detected" rather than falling back to
    the old, unreliable colour heuristic, since a silent fallback to
    the very thing being replaced would defeat the point of the fix.

    Returns (detected: bool, confidence: float).
    """
    if watch_model is None:
        return False, 0.0
    crop = _crop_wrist_band(frame, wrist_xy, scale_ref, target=infer_size)
    if crop is None:
        return False, 0.0
    try:
        results = watch_model.predict(crop, conf=conf_threshold, verbose=False)
    except Exception:
        return False, 0.0
    if not results or results[0].boxes is None:
        return False, 0.0

    best_conf = 0.0
    for box in results[0].boxes:
        try:
            if int(box.cls[0]) != class_id:
                continue
            best_conf = max(best_conf, float(box.conf[0]))
        except (IndexError, TypeError, ValueError):
            continue
    return (best_conf > 0.0), best_conf


# ─────────────────────────── close-up phone scanning ──────────────────

def merge_phone_detections(regular, scanned, iou_thresh=0.4):
    """
    Merges the whole-frame detector's phone boxes with
    PhoneProximityScanner's close-up boxes -- both shaped
    (track_id, x1, y1, x2, y2, conf) -- keeping the higher-confidence
    box wherever the same phone was caught by both passes.
    """
    combined = sorted(list(regular) + list(scanned), key=lambda d: -d[5])
    kept = []
    for d in combined:
        box = d[1:5]
        if all(box_iou(box, k[1:5]) < iou_thresh for k in kept):
            kept.append(d)
    return kept


class PhoneProximityScanner:
    """
    The phone-detection equivalent of what HeadGazeEstimator already
    does for faces: round-robin, per-student close-up scanning.

    A single whole-frame detection pass has a hard ceiling for a wide,
    distant classroom camera -- a phone that's only a handful of
    pixels in the full 1280x720 frame has already lost most of its
    detail by the time it's downscaled further for inference, and no
    amount of confidence-threshold tuning recovers detail that's
    already gone. This instead crops a generous region around each
    student's desk/lap area (not just their visible upper body -- a
    held phone is often lower than the body box) from the FULL-
    resolution frame, upscales it, and runs phone detection on that
    close-up alone, effectively giving every student a close-up look
    regardless of how small they are in the wide shot.

    Scanning everyone every frame would be far too expensive, so this
    cycles through a batch of students per call, same pattern as
    GazeTracker's face-mesh sampling, and caches each student's last
    result for GRACE_SEC so coverage stays continuous between their
    turns in the rotation.

    Deliberately NOT using .track() here -- phones don't need
    persistent identity in this app (only paper hand-offs do), so a
    plain forward pass is used, which is also cheaper.

    Even with this, a phone held below desk height, out of the
    camera's physical line of sight entirely (blocked by the desk
    itself or by other students in front), cannot be detected by this
    or any camera-based system -- that's a placement/occlusion limit,
    not a model-accuracy one. See the module-level note in eye.py's
    v5.1 changelog for what this does and doesn't solve.
    """
    GRACE_SEC = 2.0

    def __init__(self, batch_size=4, target_min=480, imgsz=384):
        self.batch_size = max(0, int(batch_size))
        self.target_min = target_min
        self.imgsz = imgsz
        self._cursor = 0
        self._cache = {}   # seat_id -> {"dets": [(x1,y1,x2,y2,conf), ...], "ts": ...}

    def set_batch_size(self, n):
        self.batch_size = max(0, int(n))

    def scan_batch(self, model, frame, seat_boxes, conf, iou, device, half, phone_cls, now):
        """seat_boxes: {seat_id: (x1,y1,x2,y2)}. No-ops if batch_size is 0 (Max Speed preset)."""
        if not seat_boxes or self.batch_size == 0:
            return
        ordered = sorted(seat_boxes.items(), key=lambda t: t[0])
        n = len(ordered)
        take = min(self.batch_size, n)
        h_frame, w_frame = frame.shape[:2]
        for i in range(take):
            sid, box = ordered[(self._cursor + i) % n]
            dets = self._scan_one(model, frame, box, conf, iou, device, half,
                                   phone_cls, h_frame, w_frame)
            self._cache[sid] = {"dets": dets, "ts": now}
        self._cursor = (self._cursor + take) % max(n, 1)

    def _scan_one(self, model, frame, box, conf, iou, device, half, phone_cls, h_frame, w_frame):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 4 or bh <= 4:
            return []
        # Expand well beyond the body box -- especially downward, to
        # cover the desk/lap area where a held phone usually actually is.
        ex1 = int(clamp(x1 - bw * 0.4, 0, w_frame))
        ex2 = int(clamp(x2 + bw * 0.4, 0, w_frame))
        ey1 = int(clamp(y1 - bh * 0.15, 0, h_frame))
        ey2 = int(clamp(y1 + bh * 2.1, 0, h_frame))
        if ex2 - ex1 < 20 or ey2 - ey1 < 20:
            return []

        crop = frame[ey1:ey2, ex1:ex2]
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(ch, cw) < self.target_min:
            scale = self.target_min / max(ch, cw)
            crop = cv2.resize(crop, (max(int(cw * scale), 1), max(int(ch * scale), 1)),
                               interpolation=cv2.INTER_LINEAR)

        try:
            res = model.predict(
                crop, classes=[phone_cls], conf=conf, iou=iou,
                imgsz=self.imgsz, device=device, half=half, verbose=False,
            )
        except Exception:
            return []

        out = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                bconf = float(b.conf[0])
                out.append((
                    int(ex1 + bx1 / scale), int(ey1 + by1 / scale),
                    int(ex1 + bx2 / scale), int(ey1 + by2 / scale), bconf,
                ))
        return out

    def current_detections(self, now):
        """All still-fresh cached detections, shaped like the regular
        detector's output (-1 track id -- phones don't carry identity)."""
        merged = []
        for entry in self._cache.values():
            if now - entry["ts"] <= self.GRACE_SEC:
                for x1, y1, x2, y2, conf in entry["dets"]:
                    merged.append((-1, x1, y1, x2, y2, conf))
        return merged


# ─────────────────────────── pose-only turn fallback ───────────────────

def fallback_turn_score(kp):
    """
    Used automatically whenever MediaPipe face-mesh data isn't
    available yet for this student this frame (not installed, model
    unavailable, or simply not this student's turn in the round-robin
    sampling). A refinement of the original nose/shoulder heuristic --
    but a deliberately conservative one. See detection_engine.py's
    fallback_turn_score change-note below for why.

    A student leaning their head down toward their own paper to write
    -- completely normal, required exam behaviour -- also shifts the
    2D nose position sideways relative to the shoulder midpoint and
    makes one ear noticeably less visible to the camera than the
    other, for pure head-geometry reasons that have nothing to do with
    a sideways glance at a neighbour. Plain 2D keypoints can't fully
    tell "tilted down to write" apart from "turned sideways to copy"
    the way real head yaw (from MediaPipe's face landmarks, which DOES
    separate yaw from pitch) can. So this fallback:
      - treats ear-visibility asymmetry as a minor corroborator only
        (small weight), since it's exactly the signal a downward tilt
        also produces,
      - requires turn_ratio AND eye_turn to roughly agree (both at
        least somewhat elevated) rather than letting either one alone
        drive the score up,
      - and is capped well below 1.0, so on its own it needs a longer
        sustained reading (via the accumulator) to ever fire -- see
        GazeTracker.score's per-source accumulation weighting in eye.py.
    None of this is needed when a fresh face-mesh sample is available;
    that path already separates "looking down" from "turned sideways"
    directly and is trusted at full strength.
    """
    if kp is None:
        return 0.0
    try:
        nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = kp[0:7]
    except (IndexError, TypeError):
        return 0.0
    if nose[2] < 0.55 or l_sh[2] < 0.35 or r_sh[2] < 0.35:
        return 0.0

    sh_cx = (l_sh[0] + r_sh[0]) / 2.0
    sh_w = abs(r_sh[0] - l_sh[0])
    if sh_w < 20:
        return 0.0

    turn_ratio = abs(nose[0] - sh_cx) / sh_w
    eye_turn = None
    if l_eye[2] > 0.35 and r_eye[2] > 0.35:
        eye_mid = (l_eye[0] + r_eye[0]) / 2.0
        eye_turn = abs(eye_mid - sh_cx) / sh_w
    ear_asym = abs(l_ear[2] - r_ear[2])

    turn_component = clamp(turn_ratio / 0.65, 0.0, 1.0)
    if eye_turn is None:
        # No usable eye-confidence data this frame -- most often this
        # means both eyes are angled away from the camera, which for a
        # classroom camera mounted high/at the front of the room
        # happens constantly during completely normal behaviour: a
        # student looking down at their own desk to write. That's the
        # overwhelmingly more likely explanation than a sideways turn
        # during an exam, so this heavily discounts the nose-offset-
        # alone reading rather than trusting it at near full strength --
        # exactly the ambiguous case a flat 2D estimate can't resolve
        # on its own.
        core = turn_component * 0.4
    else:
        eye_component = clamp(eye_turn / 0.60, 0.0, 1.0)
        # Require rough agreement instead of letting either signal
        # alone drive the score: geometric mean punishes "one signal
        # says clearly turned, the other says not really" far harder
        # than a weighted sum would.
        core = (turn_component * eye_component) ** 0.5

    ear_bonus = 0.15 * clamp(ear_asym / 0.55, 0.0, 1.0)
    score = clamp(core + ear_bonus, 0.0, 1.0)
    # Capped below 1.0 -- this signal alone should never look as
    # certain as a corroborated face-mesh reading; it needs to combine
    # with sustained duration (via the accumulator) before it can fire.
    return score * 0.80


# ─────────────────────────── MediaPipe head-pose / gaze ────────────────

_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
_MODEL_NAME = "face_landmarker.task"

# Canonical MediaPipe FaceMesh landmark indices (468-point topology).
# These specific numbers are standard/stable across the model's
# landmark ordering, not something tuned per-image.
_NOSE_TIP  = 1
_CHIN      = 152
_FOREHEAD  = 10
_FACE_L    = 234   # image-space left edge of the face
_FACE_R    = 454   # image-space right edge of the face
_L_EYE     = 33     # left eye, outer corner
_R_EYE     = 263    # right eye, outer corner
_L_MOUTH   = 61     # left mouth corner
_R_MOUTH   = 291    # right mouth corner

# Generic 3D face model (arbitrary units -- only the relative
# proportions between points matter for solvePnP, not the exact
# scale), used by _yaw_pitch_roll_solvepnp below. This is the
# standard 6-point model used across most OpenCV solvePnP head-pose
# tutorials, paired with the 6 MediaPipe indices above.
_FACE_3D_MODEL = np.array([
    (0.0,     0.0,    0.0),     # Nose tip
    (0.0,  -330.0,  -65.0),     # Chin
    (-225.0, 170.0, -135.0),    # Left eye, outer corner
    (225.0,  170.0, -135.0),    # Right eye, outer corner
    (-150.0,-150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)


def _yaw_pitch_roll_solvepnp(lm, frame_shape):
    """
    v5.2 -- REAL yaw/pitch/roll in DEGREES via OpenCV solvePnP against
    the 6-point generic 3D face model above, replacing the previous
    x/y-ratio estimate. This is what lets GazeTracker use a physically
    meaningful, explainable threshold ("turned more than 40 degrees")
    instead of a ratio cutoff tuned by feel with no obvious real-world
    interpretation.

    lm: one face's MediaPipe landmark list (result.face_landmarks[0]
        from FaceLandmarker -- each entry has normalised .x/.y in
        0..1). frame_shape: shape of whatever image lm's coordinates
        are normalised against (the head CROP passed to solvePnP
        here, not necessarily the full captured frame).

    Returns (yaw, pitch, roll, quality) in degrees; quality is the
    same "how much of the crop the face spans" framing proxy the old
    function used. Never raises -- any solvePnP failure (degenerate
    geometry, a landmark index miss) returns all zeros with
    quality 0.0, which every caller already treats as "no reading yet".

    Known quirk (documented rather than silently patched): depending
    on head geometry, cv2.decomposeProjectionMatrix's PITCH/ROLL
    values can come back with a large offset (e.g. ~180 degrees) --
    this is a well-known characteristic of this exact OpenCV
    technique. YAW (the value this app actually thresholds against
    for "turned sideways") is not affected. If you later also want to
    threshold pitch/roll directly, normalise them first, e.g.
    `pitch = pitch - 180 if pitch > 90 else pitch`.
    """
    h, w = frame_shape[:2]
    try:
        nose, chin = lm[_NOSE_TIP], lm[_CHIN]
        l_eye, r_eye = lm[_L_EYE], lm[_R_EYE]
        l_mouth, r_mouth = lm[_L_MOUTH], lm[_R_MOUTH]
        left, right = lm[_FACE_L], lm[_FACE_R]
    except IndexError:
        return 0.0, 0.0, 0.0, 0.0

    image_points = np.array([
        (nose.x * w, nose.y * h),
        (chin.x * w, chin.y * h),
        (l_eye.x * w, l_eye.y * h),
        (r_eye.x * w, r_eye.y * h),
        (l_mouth.x * w, l_mouth.y * h),
        (r_mouth.x * w, r_mouth.y * h),
    ], dtype=np.float64)

    # Approximate camera intrinsics -- fine for a relative yaw/pitch
    # reading; a full calibration isn't needed for "turned or not".
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    try:
        ok, rotation_vec, translation_vec = cv2.solvePnP(
            _FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0, 0.0
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        # decomposeProjectionMatrix returns euler angles in
        # [pitch, yaw, roll] order -- a common gotcha, easy to swap
        # yaw/pitch by accident if unpacked in a different order.
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [float(a[0]) for a in euler_angles]
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    face_w = right.x - left.x
    # Coarse "did we actually get a well-framed face" proxy from how
    # much of the crop the face spans -- not a real model confidence
    # score (FaceLandmarker doesn't expose one per-face), just a soft
    # weighting signal, unchanged from the previous version.
    quality = clamp((abs(face_w) - 0.15) / 0.35, 0.0, 1.0)
    return float(yaw), float(pitch), float(roll), float(quality)


def _gaze_from_blendshapes(categories):
    """
    Signed -1..1 'eyes turned sideways' score from ARKit-style gaze
    blendshapes, independent of head yaw -- this is what can catch a
    student glancing sideways without turning their head far enough
    to trip the head-yaw check, which a body-pose-only system has no
    way to see at all. Defensive by construction: an unrecognised or
    missing category name just contributes 0, so a MediaPipe version
    that renames/reorders blendshapes degrades this signal instead of
    raising.
    """
    if not categories:
        return 0.0
    scores = {c.category_name: c.score for c in categories if c.category_name}
    right_turn = scores.get("eyeLookOutRight", 0.0) + scores.get("eyeLookInLeft", 0.0)
    left_turn  = scores.get("eyeLookOutLeft", 0.0) + scores.get("eyeLookInRight", 0.0)
    return clamp((right_turn - left_turn) / 2.0, -1.0, 1.0)


def _crop_head(frame, box, target=320):
    """
    Crops a generously-padded, upscaled, square region around where a
    student's head should be. Two things matter here for accuracy at
    distance: (1) this crops from the FULL native-resolution captured
    frame, not the downscaled copy YOLO ran inference on, so a distant
    student's face keeps whatever real pixel detail the camera
    actually captured; (2) it's driven mainly off box HEIGHT rather
    than box WIDTH, since a person's width varies a lot with pose/arm
    position while head-height-as-a-fraction-of-body-height stays far
    more stable -- a purely width-based crop goes badly wrong on a
    wide or noisy detection box.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return None

    head_h = clamp(bh * 0.34, 24, bh)
    half_w = clamp(bw / 2.0, head_h * 0.55, head_h * 1.5)
    cx = (x1 + x2) / 2.0

    cx1 = int(clamp(cx - half_w, 0, w_frame - 1))
    cx2 = int(clamp(cx + half_w, 0, w_frame - 1))
    cy1 = int(clamp(y1 - head_h * 0.25, 0, h_frame - 1))
    cy2 = int(clamp(y1 + head_h, 0, h_frame - 1))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    square = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                 cv2.BORDER_REPLICATE)
    if side != target:
        interp = cv2.INTER_LINEAR if side < target else cv2.INTER_AREA
        square = cv2.resize(square, (target, target), interpolation=interp)
    return square


class HeadGazeEstimator:
    """
    Thin wrapper around MediaPipe's Face Landmarker. Never lets a
    missing dependency, a blocked model download, or a bad frame
    become a crash: `.available` tells the caller up front whether
    this enhancement is usable, and `.estimate()` returns None on any
    failure so callers always have a safe "no data this time" path
    (GazeTracker's pose-only fallback covers that automatically).

    The model bundle (a few MB) is cached next to the script after the
    first successful download, so after one run with internet access
    the app works fully offline from then on -- important for school
    networks that may firewall storage.googleapis.com even though
    they're fine with everything else, and for labs where only one
    machine has real internet access (copy the cached file to the
    others' `models/` folder and they'll pick it up with no download
    at all).
    """

    def __init__(self, script_dir, download_timeout=20):
        self.available = False
        self.status = "not initialised"
        self._landmarker = None
        self._mp_image_cls = None
        self._mp_image_format = None
        self._try_init(script_dir, download_timeout)

    def _try_init(self, script_dir, timeout):
        try:
            import mediapipe as mp_module
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker, FaceLandmarkerOptions, RunningMode,
            )
        except ImportError as e:
            self.status = f"mediapipe not installed ({e}) — using pose-only fallback"
            return

        model_path = os.environ.get("EYE_FACE_MODEL_PATH")
        if not model_path:
            model_dir = os.path.join(script_dir, "models")
            model_path = os.path.join(model_dir, _MODEL_NAME)
            try:
                os.makedirs(model_dir, exist_ok=True)
                if not os.path.exists(model_path):
                    self._download_model(model_path, timeout)
            except Exception as e:
                self.status = f"face-mesh model unavailable ({e}) — using pose-only fallback"
                return

        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._mp_image_cls = mp_module.Image
            self._mp_image_format = mp_module.ImageFormat.SRGB
            self.available = True
            self.status = "MediaPipe Face Landmarker active"
        except Exception as e:
            self.status = f"face-mesh init failed ({e}) — using pose-only fallback"
            self._landmarker = None

    def _download_model(self, dest_path, timeout):
        tmp_path = dest_path + ".part"
        req = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "EYE-Proctor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp_path, dest_path)

    def estimate(self, frame_bgr, box):
        """
        Returns {"yaw","pitch","gaze","quality"} or None. Never raises.
        """
        if not self.available:
            return None
        crop = _crop_head(frame_bgr, box)
        if crop is None:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
            result = self._landmarker.detect(mp_image)
        except Exception:
            return None
        if not result or not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        yaw, pitch, _roll, quality = _yaw_pitch_roll_solvepnp(lm, crop.shape)
        blend = result.face_blendshapes[0] if result.face_blendshapes else []
        gaze = _gaze_from_blendshapes(blend)
        return {"yaw": yaw, "pitch": pitch, "gaze": gaze, "quality": quality}


    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass


class GazeTracker:
    """
    Combines HeadGazeEstimator's per-student round-robin sampling with
    caching and per-seat hysteresis so eye.py just gets one clean call
    per student per frame. Round-robin (a batch of students per frame,
    cycling through the roster) instead of every-student-every-frame
    keeps a 40+ student classroom running at real-time FPS even on a
    CPU-only machine; a cached sample is trusted for GRACE_SEC before
    falling back to the pose-only estimate, so coverage stays
    responsive between a given student's samples.
    """
    GRACE_SEC = 1.2

    # _yaw_pitch_roll_solvepnp returns yaw/pitch in DEGREES (see its
    # docstring -- that's the whole point of the v5.2 solvePnP
    # rewrite, so thresholds mean something physical: "turned more
    # than ~YAW_FULL_TURN_DEG degrees"). These two constants are the
    # degree scale those readings are normalised against below. They
    # used to be 0.65 / 0.9 -- values that only make sense if yaw/
    # pitch were in RADIANS, not degrees. With real degree values,
    # abs(yaw)/0.65 saturates to 1.0 for almost any nonzero yaw
    # (anything past ~0.65 DEGREES), and abs(pitch)/0.9 crushes
    # pitch_discount to its floor for almost any nonzero pitch (past
    # ~0.9 DEGREES) -- which is every frame, since a camera mounted
    # high and angled down at the desks gives every student a
    # constant, nonzero pitch reading just from normal posture. Net
    # effect: yaw_component collapsed to a near-constant ~0.25
    # regardless of whether a student was glancing 3 degrees or fully
    # turned 60 degrees to look at a neighbour, so the facemesh path
    # could barely ever push the score over ScoreGate's enter
    # threshold on head-turn alone -- detection ended up depending
    # almost entirely on the eye-blendshape gaze signal, which is a
    # much weaker cue for a full head turn. That's the bug behind
    # COPYING missing an actual turn-to-look-at-neighbour: the numbers
    # weren't too lenient, they were in the wrong unit.
    YAW_FULL_TURN_DEG   = 38.0   # yaw_component reaches 1.0 around here
    PITCH_DISCOUNT_DEG  = 55.0   # pitch magnitude where the discount bottoms out

    def __init__(self, estimator: HeadGazeEstimator, batch_size=6):
        self.estimator = estimator
        self.batch_size = max(1, int(batch_size))
        self._cache = {}    # seat_id -> {"yaw","pitch","gaze","ts"}
        self._gates = {}    # seat_id -> ScoreGate
        self._cursor = 0

    def set_batch_size(self, n):
        self.batch_size = max(1, int(n))

    def sample_batch(self, frame, seats, now):
        """seats: list of (seat_id, box). No-ops safely if MediaPipe unavailable."""
        if not seats or not self.estimator.available:
            return
        ordered = sorted(seats, key=lambda t: t[0])
        n = len(ordered)
        take = min(self.batch_size, n)
        for i in range(take):
            seat_id, box = ordered[(self._cursor + i) % n]
            result = self.estimator.estimate(frame, box)
            if result is not None:
                self._cache[seat_id] = dict(result, ts=now)
        self._cursor = (self._cursor + take) % max(n, 1)

    def score(self, seat_id, kp, now):
        """Returns (turning_active: bool, smoothed_score: float, source: str)."""
        cached = self._cache.get(seat_id)
        if cached is not None and (now - cached["ts"]) <= self.GRACE_SEC:
            yaw, pitch, gaze = cached["yaw"], cached["pitch"], cached["gaze"]
            # A steep pitch reading (looking sharply down at the desk)
            # makes the accompanying yaw estimate itself less reliable
            # -- a foreshortened, angled-away face gives the landmark
            # fitter less to work with for left/right position, not
            # just for the (already-excluded) up/down component. This
            # matters most for a camera mounted high and at the front
            # of the room looking down across the class: a student
            # looking down at their own desk is then the expected,
            # constant, correct state, not an edge case, so yaw's
            # contribution is discounted the steeper the pitch gets
            # rather than trusted at full strength regardless.
            pitch_discount = clamp(1.0 - abs(pitch) / self.PITCH_DISCOUNT_DEG, 0.25, 1.0)
            yaw_raw = clamp(abs(yaw) / self.YAW_FULL_TURN_DEG, 0.0, 1.0)
            # The discount above is meant for the ambiguous case only:
            # a small/moderate yaw reading that could just as easily be
            # a student looking down at their own desk. It should NOT
            # apply at full strength when yaw is already large and
            # unambiguous on its own -- and in the most common real
            # violation, glancing sideways at a neighbour's desk, the
            # student's head is turned AND tilted down at the same
            # time (a neighbour's paper sits lower and off to the
            # side), so the two effects used to compound and could
            # crush an otherwise obvious look. This fades the discount
            # back out toward 1.0 as yaw_raw itself climbs, so a
            # decisive yaw reading isn't punished just for arriving
            # alongside a decisive pitch reading; it still applies at
            # full strength when yaw is small (the genuinely ambiguous
            # "just writing" case is unaffected -- yaw_raw≈0 keeps
            # effective_discount≈pitch_discount there).
            effective_discount = pitch_discount + (1.0 - pitch_discount) * yaw_raw
            yaw_component  = yaw_raw * effective_discount
            gaze_component = clamp(abs(gaze) / 0.60, 0.0, 1.0)
            raw = max(yaw_component, gaze_component * 0.9)
            source = "facemesh"
        else:
            raw = fallback_turn_score(kp)
            source = "pose"

        gate = self._gates.get(seat_id)
        if gate is None:
            # A wider gap between enter and exit than the textbook
            # default -- worth it here specifically because the cost
            # of a false "turning" read (an innocent student flagged)
            # matters more than being a little slower to catch a real one.
            gate = ScoreGate(enter=0.62, exit_=0.32, alpha=0.30)
            self._gates[seat_id] = gate
        active, smoothed = gate.update(raw)
        return active, smoothed, source

    def forget(self, seat_id):
        self._cache.pop(seat_id, None)
        self._gates.pop(seat_id, None)


# ─────────────────────────── hand-raise / finger-count signalling ──────
#
# NOT a violation detector -- this is a positive, opt-in classroom-
# management feature: a student raises a hand showing 1-4 fingers as a
# silent signal (however the teacher wants to define those, e.g.
# "1 = question, 2 = washroom, 3 = need paper, 4 = finished") instead
# of interrupting the room to ask out loud. Deliberately counts only
# the four non-thumb fingers (index/middle/ring/pinky): the thumb's
# extended/curled state depends on hand rotation in a way that needs
# more than a simple tip-vs-joint comparison to get right, and 1-4 is
# exactly the signalling range that was asked for, so it's not needed.

_FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

_HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
_HAND_MODEL_NAME = "hand_landmarker.task"


def is_hand_raised(kp):
    """
    Cheap geometric pre-filter using pose keypoints ALREADY computed
    for every student every frame -- is either wrist clearly above
    this student's own shoulder line? This decides whether it's worth
    running the far more expensive hand-landmark model at all, since
    hand-raising is a rare, deliberate event, not a continuous
    background state like head orientation. Returns the raised wrist's
    (x, y) if so (picking the higher one if both are raised), else None.
    """
    if kp is None:
        return None
    try:
        l_sh, r_sh = kp[5], kp[6]
        l_wrist, r_wrist = kp[9], kp[10]
    except (IndexError, TypeError):
        return None
    if l_sh[2] < 0.3 or r_sh[2] < 0.3:
        return None
    sh_y = (l_sh[1] + r_sh[1]) / 2.0
    sh_w = abs(r_sh[0] - l_sh[0])
    if sh_w < 20:
        return None
    margin = sh_w * 0.35   # require a clearly raised hand, not just a hovering one

    candidates = [w for w in (l_wrist, r_wrist) if w[2] > 0.4 and w[1] < sh_y - margin]
    if not candidates:
        return None
    best = min(candidates, key=lambda w: w[1])   # highest wrist wins
    return (float(best[0]), float(best[1]))


def _crop_hand(frame, wrist_xy, scale_ref, target=224):
    """
    Generous square crop around and above a wrist position (fingers on
    a raised hand extend upward from it), sized relative to scale_ref
    (e.g. shoulder width) and upscaled -- same "generous padding, let
    the specialised model localise precisely" approach as _crop_head.

    Returns (square_crop, origin_x, origin_y, side) where origin_x/y
    is the crop's top-left corner in full-frame pixel coordinates and
    side is the crop's pre-resize pixel size -- enough to map any
    landmark MediaPipe reports back onto the original frame:
        full_x = origin_x + landmark.x * side
        full_y = origin_y + landmark.y * side
    (landmark.x/y are already normalised 0..1, so the resize target
    size doesn't factor into the mapping at all.)
    Returns (None, 0, 0, 0) if the crop would be degenerate.
    """
    h_frame, w_frame = frame.shape[:2]
    wx, wy = wrist_xy
    half = clamp(scale_ref * 1.3, 40, 400)
    x1 = int(clamp(wx - half, 0, w_frame))
    x2 = int(clamp(wx + half, 0, w_frame))
    y1 = int(clamp(wy - half * 1.6, 0, h_frame))
    y2 = int(clamp(wy + half * 0.4, 0, h_frame))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, 0, 0, 0
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    square = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                 cv2.BORDER_REPLICATE)
    origin_x, origin_y = x1 - left, y1 - top
    if side != target:
        square = cv2.resize(square, (target, target), interpolation=cv2.INTER_LINEAR)
    return square, origin_x, origin_y, side


# Standard MediaPipe hand-landmark connections (wrist/finger skeleton
# edges) -- for drawing only, not used by the finger-count logic.
HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
)


def _count_raised_fingers(landmarks):
    """
    A finger counts as extended if its tip landmark sits clearly above
    (smaller y than) its own PIP-joint landmark -- the standard
    approach for an upright raised hand. The small margin requires the
    tip to be CLEARLY above the joint, not just marginally, to cut down
    on borderline/noisy counts.
    """
    count = 0
    for name, tip_idx in _FINGER_TIPS.items():
        pip_idx = _FINGER_PIPS[name]
        try:
            if landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02:
                count += 1
        except IndexError:
            continue
    return count


class HandSignalDetector:
    """
    MediaPipe Hand Landmarker wrapper, same fail-soft design as
    HeadGazeEstimator: `.available` is False if mediapipe isn't
    installed or the model can't be fetched, and every caller already
    has a "just skip this feature" path -- it never crashes the app,
    it just means hand-raise signalling silently doesn't do anything.
    """

    def __init__(self, script_dir, download_timeout=20):
        self.available = False
        self.status = "not initialised"
        self._landmarker = None
        self._mp_image_cls = None
        self._mp_image_format = None
        self._try_init(script_dir, download_timeout)

    def _try_init(self, script_dir, timeout):
        try:
            import mediapipe as mp_module
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions, RunningMode,
            )
        except ImportError as e:
            self.status = f"mediapipe not installed ({e}) — hand-raise signalling disabled"
            return

        model_path = os.environ.get("EYE_HAND_MODEL_PATH")
        if not model_path:
            model_dir = os.path.join(script_dir, "models")
            model_path = os.path.join(model_dir, _HAND_MODEL_NAME)
            try:
                os.makedirs(model_dir, exist_ok=True)
                if not os.path.exists(model_path):
                    self._download_model(model_path, timeout)
            except Exception as e:
                self.status = f"hand-landmark model unavailable ({e}) — hand-raise signalling disabled"
                return

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._mp_image_cls = mp_module.Image
            self._mp_image_format = mp_module.ImageFormat.SRGB
            self.available = True
            self.status = "MediaPipe Hand Landmarker active"
        except Exception as e:
            self.status = f"hand-landmark init failed ({e}) — hand-raise signalling disabled"
            self._landmarker = None

    def _download_model(self, dest_path, timeout):
        tmp_path = dest_path + ".part"
        req = urllib.request.Request(_HAND_MODEL_URL, headers={"User-Agent": "EYE-Proctor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp_path, dest_path)

    def analyze_hand(self, frame_bgr, wrist_xy, scale_ref):
        """
        Returns {"count": 0-4, "points": [(x,y) x21 in full-frame pixel
        coords]} or None on any failure (not installed, model
        unavailable, no hand actually found in the crop, ...). The
        mapped points are for drawing the tracked skeleton on the live
        feed; `count` is what actually drives the signal.
        """
        if not self.available:
            return None
        crop, origin_x, origin_y, side = _crop_hand(frame_bgr, wrist_xy, scale_ref)
        if crop is None:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
            result = self._landmarker.detect(mp_image)
        except Exception:
            return None
        if not result or not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        points = [(origin_x + lm.x * side, origin_y + lm.y * side) for lm in landmarks]
        return {"count": _count_raised_fingers(landmarks), "points": points}

    def count_fingers(self, frame_bgr, wrist_xy, scale_ref):
        """Backward-compatible shorthand: returns 0-4 or None."""
        result = self.analyze_hand(frame_bgr, wrist_xy, scale_ref)
        return result["count"] if result else None

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass


class HandRaiseTracker:
    """
    Debounces the raw pose-based 'wrist looks raised' signal into a
    single confirmed event per raise: the hand has to stay up for a
    short confirm window (filters a wrist passing through a raised
    position while doing something else, e.g. stretching), then won't
    re-fire again while it stays up, then applies a cooldown once
    lowered so a tired arm bobbing up and down doesn't spam signals.
    """
    CONFIRM_SEC = 0.45
    COOLDOWN_SEC = 4.0

    def __init__(self):
        self._state = {}   # seat_id -> dict

    def update(self, seat_id, raised_xy, now):
        """Returns True the instant a raise is newly confirmed (caller
        should read the finger count right away), else False."""
        st = self._state.setdefault(seat_id, {
            "up_since": None, "confirmed": False, "cooldown_until": 0.0,
        })
        if raised_xy is None:
            st["up_since"] = None
            st["confirmed"] = False
            return False

        if st["up_since"] is None:
            st["up_since"] = now
        held = now - st["up_since"]

        if not st["confirmed"] and held >= self.CONFIRM_SEC and now >= st["cooldown_until"]:
            st["confirmed"] = True
            st["cooldown_until"] = now + self.COOLDOWN_SEC
            return True
        return False

    def forget(self, seat_id):
        self._state.pop(seat_id, None)


# ─────────────────────────── paper sub-type classifier ─────────────────

def _has_spiral_binding(gray):
    """
    Looks for a band of small, regularly-spaced dark rings along one
    edge of the crop -- the visual signature of spiral/coil binding,
    which is the most reliable single classical-CV cue that separates
    a notebook from a loose sheet of paper. Restricted to a thin band
    at the left/right edge specifically (not scanned across the whole
    area) so this doesn't get confused with BUBBLE_SHEET's grid of
    circles, which is spread across the full page, not confined to one
    edge. A composition or stapled notebook with no visible spiral has
    no strong classical-CV signal to catch reliably -- this stays a
    hint, same caveat as the rest of classify_paper().
    """
    h, w = gray.shape[:2]
    band_w = max(int(w * 0.16), 6)
    if band_w * 2 >= w or h < 20:
        return False
    for band in (gray[:, :band_w], gray[:, w - band_w:]):
        if band.size == 0:
            continue
        blurred = cv2.GaussianBlur(band, (5, 5), 0)
        min_dim = max(min(blurred.shape[0], blurred.shape[1]) // 3, 3)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h // 14, 4),
            param1=60, param2=16, minRadius=2, maxRadius=min_dim,
        )
        if circles is not None and circles.shape[1] >= 5:
            return True
    return False


def _has_ruled_lines(gray):
    """
    Looks for a repeating comb of near-full-width horizontal edges --
    the visual signature of ruled/lined notebook paper. This is a
    second, independent NOTEBOOK cue alongside _has_spiral_binding,
    specifically to cover the case that function's own docstring
    already calls out as a gap: a composition, stapled, or perfect-
    bound notebook has no spiral for that function to find, but it
    almost always still has visible ruled lines, which are actually
    the more common and more reliably visible notebook cue of the two
    in a real overhead classroom crop.
    """
    h, w = gray.shape[:2]
    if h < 30 or w < 30:
        return False
    edges = cv2.Canny(gray, 40, 120)
    row_frac = edges.sum(axis=1).astype(np.float32) / (255.0 * max(w, 1))
    # A ruled line spans most of the page width, so require a decent
    # fraction of the row to look like a near-full-width edge before
    # counting that row at all -- this is what keeps this from
    # matching e.g. a block of printed text, which lights up plenty of
    # edge pixels but not in a way that spans the row width.
    strong_rows = np.where(row_frac > 0.35)[0]
    if len(strong_rows) < 4:
        return False
    gaps = np.diff(strong_rows)
    gaps = gaps[gaps > 2]   # collapse near-duplicate hits on the same physical line
    if len(gaps) < 3:
        return False
    mean_gap = float(np.mean(gaps))
    if mean_gap < 3:
        return False
    # Ruled lines are evenly spaced; a few incidentally-strong rows
    # from printed text or shadows won't be, so require the spacing
    # itself to be regular, not just present.
    spread = float(np.std(gaps)) / mean_gap
    return spread < 0.45


def classify_paper(frame, box):
    """
    Classical-CV sub-classification of a detected paper-like object.
    YOLO's COCO "book" class is the closest stock class for "a flat
    thing on a desk" -- there is no dedicated loose-paper class in
    COCO, so this is a heuristic layered on top of that detection, not
    a trained classifier:
      - YELLOW_PAD    : dominant colour in the crop is yellow.
      - NOTEBOOK      : spiral/coil binding visible along one edge, OR
                        a regular comb of ruled lines across the page
                        (catches stapled/composition notebooks with
                        no visible spiral).
      - BUBBLE_SHEET   : many small circles in a grid (Scantron-style).
      - TEST_PAPER    : default -- plain white/printed sheet, or
                        anything that didn't clearly match the others.
    Accuracy depends a lot on camera distance and lighting -- treat as
    a useful hint, not a certainty. Never raises; unexpected input
    falls back to TEST_PAPER.
    """
    try:
        x1, y1, x2, y2 = box
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "TEST_PAPER"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 50, 80), (38, 255, 255))
        yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size
        aspect = crop.shape[1] / max(crop.shape[0], 1)
        if yellow_ratio > 0.28 and 0.55 < aspect < 2.2:
            return "YELLOW_PAD"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if _has_spiral_binding(gray) or _has_ruled_lines(gray):
            return "NOTEBOOK"

        bright_ratio = float(np.count_nonzero(gray > 175)) / gray.size
        if bright_ratio > 0.42 and aspect > 0.7:
            edges = cv2.Canny(gray, 60, 160)
            if float(np.count_nonzero(edges)) / edges.size > 0.04:
                return "TEST_PAPER"

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        min_dim = max(min(crop.shape[0], crop.shape[1]) // 16, 3)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dim,
            param1=70, param2=20, minRadius=2, maxRadius=min_dim + 2,
        )
        if circles is not None and circles.shape[1] >= 8:
            return "BUBBLE_SHEET"
        if bright_ratio > 0.35:
            return "TEST_PAPER"
    except Exception:
        pass
    return "TEST_PAPER"