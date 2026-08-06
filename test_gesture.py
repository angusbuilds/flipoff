#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe>=0.10.9,<0.11", "opencv-python>=4.9", "numpy"]
# ///
"""Tests for the gesture classifier.

Two tiers, and the order matters. `fixtures.json` holds real MediaPipe world
landmarks recorded off an actual webcam -- ten genuine flip-offs and fourteen
ordinary hands -- and those are the tests that count. The synthetic hands below
them cover poses that were never recorded, but they are the junior partner on
purpose: an earlier synthetic suite passed 63 checks while the detector could
not fire at all, because the model and the metric shared the same wrong
assumption about hand geometry. Real frames cannot lie that way.

Run: ./test_gesture.py [-v]
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "flipoff", str(Path(__file__).with_name("flipoff.py")))
fo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fo)

THRESHOLD = 0.5

# Metres, roughly an adult hand: knuckle offsets from the wrist, then proximal /
# middle / distal phalanx lengths.
KNUCKLES = {
    "index":  (-0.020, 0.095, 0.0),
    "middle": (0.000, 0.098, 0.0),
    "ring":   (0.019, 0.092, 0.0),
    "pinky":  (0.036, 0.083, 0.0),
}
BONES = {
    "index":  (0.039, 0.022, 0.018),
    "middle": (0.044, 0.026, 0.019),
    "ring":   (0.041, 0.025, 0.018),
    "pinky":  (0.032, 0.018, 0.016),
}
SLOTS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

# Per-joint flexion in degrees at (mcp, pip, dip). "folded" is what a real
# flip-off's spare fingers actually do -- measured at 0.59-0.74 extension, far
# short of a clenched fist.
POSES = {
    "straight": (0, 0, 0),
    "slight":   (10, 15, 10),
    "loose":    (45, 60, 40),
    "folded":   (70, 90, 60),
    "fist":     (85, 110, 75),
}


def _rot_x(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0],
                     [0, math.cos(t), -math.sin(t)],
                     [0, math.sin(t), math.cos(t)]], dtype=np.float64)


def _rot_y(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0, math.sin(t)],
                     [0, 1, 0],
                     [-math.sin(t), 0, math.cos(t)]], dtype=np.float64)


def _rot_z(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0],
                     [math.sin(t), math.cos(t), 0],
                     [0, 0, 1]], dtype=np.float64)


def hand(poses, rotate=None, splay=None, jitter=0.0, seed=0, mirror=False):
    """21 landmarks for a hand with each finger at a named pose or angle triple."""
    splay = splay or {}
    pts = np.zeros((21, 3), dtype=np.float64)
    for finger, pose in poses.items():
        flex = POSES[pose] if isinstance(pose, str) else pose
        mcp = np.array(KNUCKLES[finger], dtype=np.float64)
        base = SLOTS[finger]
        pts[base] = mcp
        abduct = _rot_z(splay.get(finger, 0.0))
        here, total = mcp, 0.0
        for k, (f, length) in enumerate(zip(flex, BONES[finger])):
            total += f
            here = here + length * (abduct @ (_rot_x(total) @ np.array([0.0, 1.0, 0.0])))
            pts[base + 1 + k] = here
    pts[1:5] = [(-0.030, 0.035, 0.004), (-0.045, 0.055, 0.008),
                (-0.040, 0.072, 0.010), (-0.032, 0.084, 0.012)]
    if mirror:
        pts[:, 0] *= -1.0
    if rotate is not None:
        pts = pts @ rotate.T
    if jitter:
        pts = pts + np.random.default_rng(seed).normal(0, jitter, pts.shape)
    return pts.astype(np.float32)


def fingers(index, middle, ring, pinky):
    return {"index": index, "middle": middle, "ring": ring, "pinky": pinky}


FLIP = fingers("folded", "straight", "folded", "folded")

CASES = [
    ("flip off, textbook",        FLIP, {}, True),
    ("flip off, bent middle",     fingers("folded", "slight", "folded", "fist"), {}, True),
    ("flip off, tight fist",      fingers("fist", "straight", "fist", "fist"), {}, True),
    ("flip off, hyperextended",   fingers("folded", (-25, -5, -5), "folded", "folded"), {}, True),
    ("flip off, at camera",       FLIP, {"rotate": _rot_x(-85)}, True),
    ("flip off, upside down",     FLIP, {"rotate": _rot_x(180)}, True),
    ("flip off, rolled 60deg",    FLIP, {"rotate": _rot_y(60)}, True),
    ("flip off, angled 45deg",    FLIP, {"splay": {"middle": 45}}, True),
    ("flip off, splayed folders", FLIP, {"splay": {"index": -18, "pinky": 22}}, True),
    ("flip off, jittered",        FLIP, {"jitter": 0.002}, True),
    ("flip off, left hand",       FLIP, {"mirror": True}, True),

    # A flip-off nobody committed to. Geometrically identical to the hand-to-face
    # poses below, so it has to lose -- the folded fingers are what make the
    # gesture legible at all.
    ("lazy flip, nothing folded", fingers("loose", "straight", "loose", "loose"), {}, False),

    # Hand-to-face. Each has a near-straight middle finger with the others merely
    # relaxed; every one of these fired before the folded bands were tightened.
    ("nose scratch with middle",
     fingers((35, 50, 30), (10, 15, 10), (35, 50, 30), (40, 55, 35)), {}, False),
    ("push glasses up with middle",
     fingers((40, 55, 35), (5, 10, 5), (40, 55, 35), (45, 60, 40)), {}, False),
    ("forehead scratch",
     fingers((30, 45, 25), (0, 5, 5), (35, 50, 30), (40, 55, 35)), {}, False),
    ("middle finger on trackpad",
     fingers((25, 40, 20), (5, 10, 5), (30, 45, 25), (35, 50, 30)), {}, False),
    ("chin rest, others relaxed",
     fingers((45, 60, 40), (15, 20, 10), (45, 60, 40), (50, 65, 45)), {}, False),
    ("hands loose over keyboard",
     fingers((30, 45, 25), (15, 20, 10), (35, 50, 30), (40, 55, 35)), {}, False),
    ("cupped around a mug",
     fingers((45, 60, 40), (40, 55, 35), (45, 60, 40), (45, 60, 40)), {}, False),

    ("open palm",           fingers("straight", "straight", "straight", "straight"), {}, False),
    ("open palm, at camera", fingers("straight", "straight", "straight", "straight"),
                            {"rotate": _rot_x(-85)}, False),
    ("fist",                fingers("fist", "fist", "fist", "fist"), {}, False),
    ("peace sign",          fingers("straight", "straight", "folded", "folded"), {}, False),
    ("pointing",            fingers("straight", "folded", "folded", "folded"), {}, False),
    ("pinky out",           fingers("folded", "folded", "folded", "straight"), {}, False),
    ("middle + ring up",    fingers("folded", "straight", "straight", "folded"), {}, False),
    ("rock horns",          fingers("straight", "folded", "folded", "straight"), {}, False),
    ("relaxed half-open",   fingers("slight", "slight", "slight", "slight"), {}, False),
    ("three fingers up",    fingers("straight", "straight", "straight", "folded"), {}, False),
]


def _check(name, ok, detail="", failures=None):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{detail}")
    return 0 if ok else 1


def real_frame_tests(verbose):
    """The tests that actually matter: landmarks recorded off a real webcam."""
    path = Path(__file__).with_name("fixtures.json")
    if not path.exists():
        print("SKIP  fixtures.json missing - real-frame tests not run")
        return 0
    data = json.loads(path.read_text())
    failures = 0

    for label, frames, want_fire in (("real flip-off", data["gesture"], True),
                                     ("real ordinary hand", data["not_gesture"], False)):
        fired = 0
        worst = None
        for f in frames:
            score, ext = fo.score_hand(np.array(f, dtype=np.float32))
            hit = score >= THRESHOLD
            fired += hit
            if hit != want_fire and worst is None:
                worst = (score, ext)
        ok = (fired == len(frames)) if want_fire else (fired == 0)
        failures += _check(
            f"{label}: {fired}/{len(frames)} fire "
            f"(want {'all' if want_fire else 'none'})",
            ok,
            f"   {fo._ext_str(worst[1])}" if worst and verbose else "")
    return failures


# Poses this classifier knowingly cannot separate. Printed, never silently
# dropped: a limitation you can see beats a test that was quietly deleted.
LIMITATIONS = [
    ("middle straight but folded 90deg at the knuckle",
     fingers("folded", (90, 0, 0), "folded", "folded"),
     "no measure separates it. Real flip-offs run 68-82deg at the knuckle and "
     "ordinary hands 48-127deg, so the ranges overlap; every threshold that "
     "rejects this pose also rejects genuine gestures."),
    ("a hand resting under the chin, three fingers folded, middle out",
     fingers((60, 80, 50), (20, 25, 15), (55, 75, 45), (50, 70, 40)),
     "geometrically this IS the gesture. Sit like that and it will fire."),
]


def limitation_report():
    print("(known, unfixable without a different sensor)")
    for name, poses, why in LIMITATIONS:
        score, _ = fo.score_hand(hand(poses))
        print(f"  scores {score:.2f}  {name}\n      {why}")
    return 0


def synthetic_tests(verbose):
    failures = 0
    for name, poses, kwargs, want in CASES:
        score, ext = fo.score_hand(hand(poses, **kwargs))
        ok = (score >= THRESHOLD) == want
        failures += _check(
            f"{name:29} score={score:.2f} want={'fire' if want else 'quiet'}",
            ok, f"   [{fo._ext_str(ext)}]" if (verbose or not ok) else "")
    return failures


class _LM:
    def __init__(self, x, y, presence=None, visibility=None):
        self.x, self.y, self.z = x, y, 0.0
        self.presence, self.visibility = presence, visibility


def usable_tests():
    """The pre-filter that keeps half-seen hands from voting at all."""
    failures = 0
    centred = [_LM(0.5, 0.5) for _ in range(21)]
    failures += _check("accepts a centred, confident hand",
                       fo.hand_is_usable(centred, 0.95))
    failures += _check("rejects a low-confidence detection",
                       not fo.hand_is_usable(centred, 0.3))

    clipped = [_LM(0.5, 0.5) for _ in range(21)]
    clipped[fo.RING[3]] = _LM(-0.15, 0.5)
    failures += _check("rejects a hand clipped by the frame",
                       not fo.hand_is_usable(clipped, 0.95))

    edge = [_LM(0.5, 0.5) for _ in range(21)]
    edge[fo.PINKY[3]] = _LM(1.01, 0.99)
    failures += _check("tolerates landmarks a hair past the edge",
                       fo.hand_is_usable(edge, 0.95))

    absent = [_LM(0.5, 0.5) for _ in range(21)]
    absent[fo.INDEX[3]] = _LM(0.5, 0.5, presence=0.1)
    failures += _check("rejects a landmark the model says isn't there",
                       not fo.hand_is_usable(absent, 0.95))
    failures += _check("ignores presence/visibility when unpopulated",
                       fo.hand_is_usable(centred, 0.9))
    return failures


def detector_tests():
    """Debounce: hold time, dropout tolerance, re-arming."""
    failures = 0

    det = fo.Detector(THRESHOLD, hold=0.5)
    failures += _check("fires once while the gesture is held",
                       sum(det.update(1.0, t / 30) for t in range(60)) == 1)

    det = fo.Detector(THRESHOLD, hold=0.5)
    failures += _check("ignores a gesture flashed for 0.27s",
                       sum(det.update(1.0, t / 30) for t in range(8)) == 0)

    det = fo.Detector(THRESHOLD, hold=0.5)
    failures += _check("survives a single dropped frame",
                       sum(det.update(0.0 if t == 12 else 1.0, t / 30)
                           for t in range(60)) == 1)

    det = fo.Detector(THRESHOLD, hold=0.5)
    for t in range(20):
        det.update(1.0, t / 30)
    for t in range(20, 30):
        det.update(0.0, t / 30)
    failures += _check("clears the hold once the hand is gone", det.since is None)

    det = fo.Detector(THRESHOLD, hold=0.3)
    first = sum(det.update(1.0, t / 30) for t in range(200))
    second = sum(det.update(0.0, (200 + t) / 30) for t in range(20))
    third = sum(det.update(1.0, (220 + t) / 30) for t in range(60))
    failures += _check("needs a release before firing again",
                       (first, second, third) == (1, 0, 1))

    det = fo.Detector(THRESHOLD, hold=0.3)
    failures += _check("stays quiet below threshold",
                       sum(det.update(0.45, t / 30) for t in range(120)) == 0)

    det = fo.Detector(THRESHOLD, hold=0.3)
    sum(det.update(1.0, t / 30) for t in range(60))
    det.reset()
    failures += _check("re-fires cleanly after reset",
                       sum(det.update(1.0, (60 + t) / 30) for t in range(60)) == 1)
    return failures


def noise_tests():
    """End-to-end under landmark noise, which is the only rate that matters.

    A single frame's score is not the question -- the Detector smooths a run of
    them. Poses are held for 2s at 30fps with fresh noise every frame. MediaPipe's
    world-landmark error runs about 2-3mm on a well-lit hand, so 4mm is already
    pessimistic.
    """
    failures = 0
    trials, frames, fps = 40, 60, 30

    def fire_rate(poses, sigma_mm):
        hits = 0
        for t in range(trials):
            det = fo.Detector(THRESHOLD, hold=0.2)
            for f in range(frames):
                score, _ = fo.score_hand(
                    hand(poses, jitter=sigma_mm / 1000, seed=t * 1000 + f))
                if det.update(score, f / fps):
                    hits += 1
                    break
        return hits / trials

    for label, poses in (
        ("textbook flip-off", FLIP),
        ("hyperextended middle", fingers("folded", (-25, -5, -5), "folded", "folded")),
    ):
        r = fire_rate(poses, 3)
        failures += _check(f"fires at 3mm noise: {label:24} {r:5.0%} (>=85%)",
                           r >= 0.85)

    for label, poses in (
        ("push glasses up",
         fingers((40, 55, 35), (5, 10, 5), (40, 55, 35), (45, 60, 40))),
        ("nose scratch",
         fingers((35, 50, 30), (10, 15, 10), (35, 50, 30), (40, 55, 35))),
        ("open palm", fingers("straight", "straight", "straight", "straight")),
    ):
        r = fire_rate(poses, 4)
        failures += _check(f"silent at 4mm noise: {label:24} {r:5.0%} (0%)",
                           r == 0.0)
    return failures


def main():
    verbose = "-v" in sys.argv
    total = 0
    for title, fn in (("REAL RECORDED FRAMES", lambda: real_frame_tests(verbose)),
                      ("SYNTHETIC POSES", lambda: synthetic_tests(verbose)),
                      ("HAND USABILITY", usable_tests),
                      ("DEBOUNCE", detector_tests),
                      ("NOISE", noise_tests),
                      ("KNOWN LIMITATIONS", limitation_report)):
        print(f"\n--- {title} ---")
        total += fn()
    print(f"\n{'FAILED: ' + str(total) if total else 'ALL PASS'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
