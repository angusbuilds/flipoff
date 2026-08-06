#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "mediapipe>=0.10.9,<0.11",
#     "opencv-python>=4.9",
#     "numpy",
# ]
# ///
"""Flip off your webcam; it texts for you.

The camera is read continuously, whatever app you are in. When the gesture
lands it says so out loud, then sends -- either to a pinned recipient (--to) or
into whichever iMessage conversation is open.

Headless by default: a preview window would make *this* process frontmost,
which defeats the whole point. Use --tune to watch the score live.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # before mediapipe pulls in glog

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker"
             "/hand_landmarker/float16/1/hand_landmarker.task")

WRIST = 0
# (mcp, pip, dip, tip). The thumb is deliberately absent: people tuck it, splay
# it, or wrap it over the index, and none of that changes what the gesture means.
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (INDEX, MIDDLE, RING, PINKY)
FOLDERS = (INDEX, RING, PINKY)

# Extension ratios, calibrated against real recorded hands rather than synthetic
# geometry. Each pair is (score 0 here, score 1 here), so half-folded fingers
# degrade smoothly instead of falling off a cliff -- real flip-offs are sloppy.
#
# The numbers that matter, measured from an actual flip-off: the middle finger
# reads 0.93-0.97, while the folded fingers only reach 0.59-0.74. They are
# nowhere near the ~0.35 of a clenched fist -- nobody actually makes a fist to
# flip someone off, and assuming they did is what made this reject everything.
MIDDLE_EXTENDED = (0.80, 0.90)
OTHER_FOLDED = (0.80, 0.70)     # descending: smaller ratio means more folded

# The ring finger gets its own looser band. Its tendons share a sheath with the
# middle finger's, so plenty of people physically cannot fold it much while the
# middle stands up -- holding it to the index's standard loses real gestures.
# Safe to loosen because the index and pinky still have to be properly folded.
RING_FOLDED = (0.82, 0.72)
FOLD_BAND = {INDEX: OTHER_FOLDED, RING: RING_FOLDED, PINKY: OTHER_FOLDED}

# Straight-but-folded-at-the-knuckle reads as extended by ratio alone; this
# catches it. Measured, not assumed: on real gestures this runs 1.21-1.64, well
# short of the ~2.0 the geometry suggests, because MediaPipe puts the wrist
# landmark at the base of the palm so the knuckle starts out far from it. Set
# at 1.55 it rejected 60 of 74 genuine frames.
MIN_MIDDLE_REACH = 1.15

# A curled fingertip sits far closer to the wrist than an extended one. Cheap,
# highly discriminative, and only meaningful because these are true 3D metric
# coordinates -- in 2D a finger aimed at the lens fails this badly.
REACH_MARGIN = 1.05

# A hand running off the edge of the frame has landmarks the model extrapolated
# rather than saw, and the world coordinates derived from them are fiction --
# a clipped ring finger can read as folded purely because it was guessed at.
# Cheaper to ignore the frame than to score garbage.
FRAME_EDGE = 0.02
MIN_CONFIDENCE = 0.6

CONNECTIONS = (  # for --tune only
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


def extension(pts, finger):
    """Straight-line knuckle-to-tip over the bone path: 1.0 straight, ~0.35 fisted.

    This replaced an angle-chain measure that needed a palm-frame reference
    vector. On real hands that reference was off by enough to add ~100 degrees
    to every finger, so a genuinely extended middle finger scored as bent and
    nothing ever fired. A ratio of two lengths needs no reference at all, which
    makes it immune to hand orientation and to metacarpal geometry the synthetic
    model got wrong.
    """
    j = pts[list(finger)]
    path = sum(float(np.linalg.norm(j[k + 1] - j[k])) for k in range(3))
    if path < 1e-9:
        return 0.0
    return float(np.linalg.norm(j[3] - j[0])) / path


def middle_reach(pts):
    """Wrist-to-tip over wrist-to-knuckle for the middle finger, ~2.0 extended.

    `extension` alone cannot see a finger that is straight but folded down at
    the knuckle -- the bones stay in line, so the ratio stays near 1.0. This
    catches that, and is scale-free so it holds for any hand size.
    """
    base = float(np.linalg.norm(pts[MIDDLE[0]] - pts[WRIST]))
    if base < 1e-9:
        return 0.0
    return float(np.linalg.norm(pts[MIDDLE[3]] - pts[WRIST])) / base


def _sat(value, low, high):
    """Ramp `value` to 0..1, hitting 0 at `low` and 1 at `high`."""
    return max(0.0, min(1.0, (value - low) / (high - low)))


def score_hand(pts):
    """0..1 confidence that this hand is flipping the camera off.

    The score is the *weakest* satisfied constraint, so one straight ring
    finger sinks the whole thing -- exactly what we want, since the difference
    between this gesture and an open palm is the other three fingers.
    """
    ext = {f: extension(pts, f) for f in FINGERS}

    # A finger folded at the knuckle but straight along its own bones fools the
    # extension ratio; this does not.
    if middle_reach(pts) < MIN_MIDDLE_REACH:
        return 0.0, ext

    reach = np.linalg.norm(pts[MIDDLE[3]] - pts[WRIST])
    for f in FOLDERS:
        if reach < REACH_MARGIN * np.linalg.norm(pts[f[3]] - pts[WRIST]):
            return 0.0, ext

    parts = [_sat(ext[MIDDLE], *MIDDLE_EXTENDED)]
    parts += [_sat(ext[f], *FOLD_BAND[f]) for f in FOLDERS]
    return min(parts), ext


def _ext_str(ext):
    if not ext:
        return ""
    return "  ".join(f"{n}={ext[f]:.2f}" for n, f in
                     (("idx", INDEX), ("mid", MIDDLE), ("rng", RING), ("pky", PINKY)))


def hand_is_usable(norm, confidence):
    """Reject hands the tracker only half-saw, before their numbers get a vote.

    `presence` and `visibility` are populated on some builds and left at None on
    others, so they're used only when they're actually there.
    """
    if confidence < MIN_CONFIDENCE:
        return False
    for p in norm:
        if not (-FRAME_EDGE <= p.x <= 1 + FRAME_EDGE
                and -FRAME_EDGE <= p.y <= 1 + FRAME_EDGE):
            return False
        if p.presence is not None and p.presence < 0.5:
            return False
        if p.visibility is not None and p.visibility < 0.5:
            return False
    return True


class Detector:
    """Turns a stream of per-frame scores into one debounced trigger.

    Smoothing rides out the odd dropped frame from tracking jitter; the hold
    keeps a hand that merely passes through the pose from firing; and `armed`
    forces you to drop the gesture before it can go again.
    """

    def __init__(self, threshold, hold, smoothing=0.65):
        self.threshold = threshold
        self.hold = hold
        self.smoothing = smoothing
        self.smooth = 0.0
        self.since = None
        self.armed = True

    def reset(self):
        # armed too: without it, leaving Messages mid-gesture wedges the detector
        # shut until a low-scoring frame happens to arrive. Rebuilding `smooth`
        # from zero plus the hold is protection enough on its own.
        self.smooth = 0.0
        self.since = None
        self.armed = True

    def update(self, raw, now):
        self.smooth += self.smoothing * (raw - self.smooth)
        # Releasing at a lower bar than triggering stops the hold timer from
        # restarting every time the score wobbles across the line.
        if self.smooth < self.threshold * 0.6:
            self.since = None
            self.armed = True
            return False
        if self.smooth < self.threshold:
            return False
        if self.since is None:
            self.since = now
            return False
        if now - self.since < self.hold or not self.armed:
            return False
        self.armed = False
        return True


class _Failed:
    """Stands in for a run that never returned, so callers can treat a hung
    osascript exactly like a failed one."""
    returncode, stdout, stderr = 1, "", "osascript timed out (permission prompt?)"


def _osascript(script):
    # An ungranted Automation prompt is a modal dialog. Under launchd there is
    # nobody to click it, so osascript blocks until the timeout -- which used to
    # raise straight through main() and kill the process, over and over, because
    # KeepAlive restarted it into the same wall.
    try:
        return subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return _Failed()


# Messages must be the app you're actually looking at AND have a conversation
# open -- with the app focused but no thread selected there is nowhere for the
# text to land. The window title is the conversation name, which doubles as a
# way to log who is about to get it. One osascript round-trip for all of it.
GATE = """
tell application "System Events"
	set fg to name of first application process whose frontmost is true
end tell
if fg is not "Messages" then return "NOTFRONT"
tell application "Messages"
	if (count of windows) is 0 then return "NOWINDOW"
	set t to name of window 1
	if t is missing value then return "NOCHAT"
	if t is "" then return "NOCHAT"
	if t is "Messages" then return "NOCHAT"
	return t
end tell
"""


_gate_complained = False


def messages_target():
    """Name of the open conversation, or None if we shouldn't be typing at all."""
    global _gate_complained
    r = _osascript(GATE)
    if r.returncode != 0:
        # Denied Automation permission fails every single poll. Treated as just
        # another "not in a conversation" this would run forever detecting
        # nothing, looking identical to a working install.
        if not _gate_complained:
            print(f"cannot read Messages: {r.stderr.strip()}\n"
                  "Grant Automation access (System Settings > Privacy & Security "
                  "> Automation), or nothing will ever fire.",
                  file=sys.stderr, flush=True)
            _gate_complained = True
        return None
    out = r.stdout.strip()
    return None if out in ("", "NOTFRONT", "NOWINDOW", "NOCHAT") else out


def accessibility_ok():
    return _osascript(
        'tell application "System Events" to return UI elements enabled'
    ).stdout.strip() == "true"


def send(text):
    """Type into the focused conversation and press return.

    Messages exposes no "currently selected chat" in its AppleScript dictionary,
    so keystroking the front window is the only way to hit the thread that's
    actually open.
    """
    # Pasted, not typed. `keystroke` cannot produce astral-plane characters --
    # the bold capitals live at U+1D5D4 and up, and typing them emits a run of
    # junk like "aaaaaaa" instead. The clipboard carries any Unicode intact.
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    r = _osascript(
        "set saved to \"\"\n"
        "try\n\tset saved to the clipboard as text\nend try\n"
        f'set the clipboard to "{escaped}"\n'
        'tell application "System Events" to tell process "Messages"\n'
        '\tkeystroke "v" using command down\n'
        "\tdelay 0.15\n"
        "\tkey code 36\n"
        "end tell\n"
        "delay 0.25\n"
        "try\n\tset the clipboard to saved\nend try"
    )
    if r.returncode != 0:
        print(f"  send failed: {r.stderr.strip()}", file=sys.stderr, flush=True)
        return False
    return True


# Unicode "mathematical sans-serif bold" capitals. iMessage has no markdown, so
# this is the only way to make the text itself render bold -- and being ordinary
# Unicode it survives SMS, Android and notification previews alike. It is also
# why the fallback send pastes instead of typing: `keystroke` cannot emit these.
_BOLD_A = 0x1D5D4
_BOLD_0 = 0x1D7EC


def boldize(text):
    out = []
    for ch in text.upper():
        if "A" <= ch <= "Z":
            out.append(chr(_BOLD_A + ord(ch) - ord("A")))
        elif "0" <= ch <= "9":
            out.append(chr(_BOLD_0 + ord(ch) - ord("0")))
        else:
            out.append(ch)
    return "".join(out)


# iMessage plays a full-screen effect when the message contains one of these
# phrases. It is keyword detection on the RECEIVING device, which is why it
# works from a Mac at all -- the compose-side effects picker is iOS-only, so
# there is no menu here to automate.
EFFECT_TRIGGERS = {
    "fireworks": "Happy New Year",
    "confetti": "Congratulations",
    "balloons": "Happy Birthday",
    "lasers": "Pew pew",
    "none": "",
}


def speak(text, voice="Cellos"):
    """Sing it out loud the instant the gesture lands.

    Fires on detection rather than on a successful send, so it is also the
    fastest honest feedback there is: if you hear it, it saw you.
    """
    # Rate 220 was fast enough to slur the words into noise; the singing voices
    # also carry their own tempo, so leave the rate alone entirely.
    try:
        subprocess.Popen(["say", "-v", voice, text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def resolve_handle(who):
    """A contact name, or a handle already -- either way, something addressable."""
    if "@" in who or (any(c.isdigit() for c in who) and " " not in who):
        return who
    esc = who.replace('"', '\\"')
    r = _osascript(
        'tell application "Messages"\n'
        "\trepeat with p in participants\n"
        '\t\tset nm to ""\n'
        '\t\tset hd to ""\n'
        "\t\ttry\n\t\t\tset nm to name of p\n\t\tend try\n"
        "\t\ttry\n\t\t\tset hd to handle of p\n\t\tend try\n"
        f'\t\tif nm contains "{esc}" and hd is not "" then return hd\n'
        "\tend repeat\n"
        '\treturn ""\n'
        "end tell"
    )
    return r.stdout.strip() or None


def send_direct(handle, text):
    """Address the recipient by handle instead of typing into the front window.

    Sturdier than keystrokes in every way: it doesn't care which app is focused,
    doesn't need the message box to have keyboard focus, and doesn't need
    Accessibility at all.
    """
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    r = _osascript(
        'tell application "Messages"\n'
        "\tset svc to 1st account whose service type = iMessage\n"
        f'\tsend "{esc}" to participant "{handle}" of svc\n'
        "end tell"
    )
    if r.returncode != 0:
        print(f"  send failed: {r.stderr.strip()}", file=sys.stderr, flush=True)
    return r.returncode == 0


def ensure_model():
    if MODEL.exists():
        return
    print(f"fetching hand model -> {MODEL.name}", flush=True)
    tmp = MODEL.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(MODEL)


def _draw(frame, hands, raw, det, threshold):
    h, w = frame.shape[:2]
    for hand in hands:
        px = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, px[a], px[b], (200, 200, 200), 2)
        for i, p in enumerate(px):
            hot = i in MIDDLE
            cv2.circle(frame, p, 5 if hot else 3,
                       (0, 220, 255) if hot else (120, 120, 120), -1)
    colour = (0, 200, 0) if det.smooth >= threshold else (60, 60, 220)
    cv2.putText(frame, f"raw {raw:.2f}   smooth {det.smooth:.2f}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.rectangle(frame, (12, 44), (12 + int(300 * det.smooth), 60), colour, -1)
    cv2.rectangle(frame, (12, 44), (312, 60), (200, 200, 200), 1)
    x = 12 + int(300 * threshold)
    cv2.line(frame, (x, 40), (x, 64), (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Flip off the camera, send a text.")
    ap.add_argument("-m", "--message", default="fuck you")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="0..1 confidence needed to fire; lower is twitchier "
                         "(default: 0.5)")
    ap.add_argument("--hold", type=float, default=0.2,
                    help="seconds the gesture must be held (default: 0.2)")
    ap.add_argument("--cooldown", type=float, default=8.0,
                    help="seconds before it can fire again (default: 8)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--test", action="store_true",
                    help="log detections, never send, ignore which app is frontmost")
    ap.add_argument("--tune", action="store_true",
                    help="camera window with live scores; implies --test")
    ap.add_argument("--effect", choices=tuple(EFFECT_TRIGGERS), default="fireworks",
                    help="iMessage screen effect to play on their device "
                         "(default: fireworks)")
    ap.add_argument("--bold", action=argparse.BooleanOptionalAction, default=True,
                    help="send the message in bold uppercase (default: on)")
    ap.add_argument("--hand", choices=("right", "left", "any"), default="right",
                    help="which hand counts (default: right) -- ignoring the "
                         "other one removes a whole class of false positives")
    ap.add_argument("--say", action=argparse.BooleanOptionalAction, default=True,
                    help="sing the message out loud when it fires (default: on)")
    ap.add_argument("--voice", default="Cellos",
                    help="voice for the spoken effect; the singing ones are "
                         "Cellos, Good News, Bells, Organ, Bad News, Boing "
                         "(default: Cellos)")
    ap.add_argument("--to", metavar="NAME",
                    help="always text this person (name or handle) instead of "
                         "whatever conversation happens to be open; needs no "
                         "window focus and no Accessibility")
    ap.add_argument("--record", metavar="FILE",
                    help="append every scored frame as JSON -- real landmarks "
                         "from your own hand, for tuning against instead of "
                         "synthetic geometry")
    args = ap.parse_args()
    if args.tune:
        args.test = True

    ensure_model()

    spoken = args.message
    if args.bold:
        args.message = boldize(args.message)
    trigger = EFFECT_TRIGGERS[args.effect]
    if trigger:
        args.message = f"{args.message} {trigger}"

    handle = None
    if args.to:
        handle = resolve_handle(args.to)
        if not handle:
            sys.exit(f"no Messages contact matching {args.to!r} - "
                     "use the exact name as it appears in Messages, or a number")
        print(f"pinned recipient: {args.to} <{handle}>", flush=True)

    # Fail here rather than after a successful gesture that silently goes nowhere.
    # Direct send addresses the person, so it needs no keystrokes at all.
    if not args.test and not handle and not accessibility_ok():
        sys.exit("Accessibility is off, so nothing can be typed into Messages.\n"
                 "System Settings > Privacy & Security > Accessibility, tick your "
                 "terminal, then rerun.")

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        sys.exit(f"could not open camera {args.camera} "
                 "(check System Settings > Privacy & Security > Camera)")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,  # either hand, or one of two in frame
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )
    det = Detector(args.threshold, args.hold)
    record = open(args.record, "a", buffering=1) if args.record else None

    print(f"watching. message: {args.message!r}"
          + ("  [TEST MODE - nothing will send]" if args.test else ""),
          flush=True)
    print("ctrl-c to stop", flush=True)

    last_sent = 0.0
    target = None
    gate_checked = 0.0
    stamp_ms = 0
    episode_seen = 0.0        # when a hand was last visible
    episode_best = 0.0        # best raw score during this sighting
    episode_detail = None
    episode_logged = 0.0
    episode_hand = "?"

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.1)
                continue

            now = time.monotonic()

            # The camera is read every frame, always, whatever app you're in.
            # This used to skip tracking entirely unless Messages was frontmost,
            # which saved CPU but meant switching to Messages and immediately
            # flipping off got missed: the gate was only polled once a second and
            # the tracker had to re-acquire from cold. Whether you're *in* a
            # conversation is now asked once, at the moment it fires.
            frame = cv2.flip(frame, 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stamp_ms = max(stamp_ms + 1, int(now * 1000))  # must strictly increase
            result = landmarker.detect_for_video(image, stamp_ms)

            # World landmarks are metric 3D with the origin at the hand's centre,
            # so curl angles hold up when the finger is aimed down the lens and
            # its 2D projection collapses to nothing.
            raw, best, raw_hand = 0.0, None, "?"
            for i, world in enumerate(result.hand_world_landmarks):
                norm = result.hand_landmarks[i]
                conf = result.handedness[i][0].score if result.handedness else 1.0
                # The frame is mirrored before detection, which is the selfie
                # orientation MediaPipe assumes, so its label needs no swapping.
                # Verified against a real right hand rather than reasoned about.
                if args.hand != "any" and result.handedness:
                    if result.handedness[i][0].category_name.lower() != args.hand:
                        continue
                if not hand_is_usable(norm, conf):
                    continue
                pts = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32)
                s, curls = score_hand(pts)
                if s >= raw:
                    raw, best = s, curls
                    raw_hand = (result.handedness[i][0].category_name
                                if result.handedness else "?")
                if record:
                    record.write(json.dumps({
                        "t": round(now, 3), "score": round(s, 4), "conf": round(conf, 3),
                        "curls": {n: round(curls[f], 1) for n, f in
                                  (("index", INDEX), ("middle", MIDDLE),
                                   ("ring", RING), ("pinky", PINKY))},
                        "world": [[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                                  for p in world],
                    }) + "\n")

            fired = det.update(raw, now)

            # Every attempt gets logged, fired or not. Without this a gesture
            # that scores 0.44 and one the camera never saw look identical from
            # outside -- silence -- and there is no way to tell which you had.
            if best is not None:
                if raw >= episode_best:
                    episode_best, episode_detail = raw, best
                    episode_hand = raw_hand
                episode_seen = now
            elif episode_seen and now - episode_seen > 0.6:
                if not fired and now - episode_logged > 2.0:
                    episode_logged = now
                    if episode_best < args.threshold:
                        verdict = "below threshold"
                    elif now - last_sent < args.cooldown:
                        verdict = (f"cooling down, {args.cooldown - (now - last_sent):.0f}s "
                                   "left")
                    else:
                        verdict = "not held long enough"
                    print(f"[{time.strftime('%H:%M:%S')}] saw {episode_hand} hand, "
                          f"best {episode_best:.2f} (need {args.threshold:.2f}) "
                          f"- {verdict}  {_ext_str(episode_detail)}", flush=True)
                episode_seen, episode_best, episode_detail = 0.0, 0.0, None

            if args.tune:
                cv2.imshow("flipoff --tune",
                           _draw(frame, result.hand_landmarks, raw, det,
                                 args.threshold))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if fired and now - last_sent >= args.cooldown:
                clock = time.strftime("%H:%M:%S")
                detail = _ext_str(best)
                if args.say:
                    speak(spoken, args.voice)
                print(f"[{clock}] FIRED on {raw_hand} hand", flush=True)
                if args.test:
                    last_sent = now
                    print(f"[{clock}] detected {det.smooth:.2f}  {detail}",
                          flush=True)
                    continue
                # Re-confirm rather than trusting the cache: the gesture took
                # 0.7s to complete and you may have switched away inside it.
                if handle:
                    if send_direct(handle, args.message):
                        last_sent = now
                        print(f"[{clock}] -> {args.to}: {args.message!r}  "
                              f"({det.smooth:.2f})", flush=True)
                    continue
                now_target = messages_target()
                if not now_target:
                    print(f"[{clock}] detected, but no conversation open - skipped",
                          flush=True)
                    continue
                # Prefer addressing the selected conversation by handle. The
                # window title is the contact's name, so it resolves the same way
                # --to does -- which means no dependence on the message box
                # actually holding keyboard focus. Pasting is the fallback for
                # group chats and titles that match no single participant.
                to_handle = resolve_handle(now_target)
                if to_handle and send_direct(to_handle, args.message):
                    last_sent = now
                    print(f"[{clock}] -> {now_target}: {args.message!r}  "
                          f"({det.smooth:.2f})", flush=True)
                elif send(args.message):
                    # Only a real send starts the cooldown. Charging it for a
                    # skip would mute the next 8 seconds over a message that
                    # was never actually sent.
                    last_sent = now
                    print(f"[{clock}] -> {now_target}: {args.message!r}  "
                          f"({det.smooth:.2f})", flush=True)

            time.sleep(0.02)  # cap around 30fps; the accurate model isn't free
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cam.release()
        landmarker.close()
        if record:
            record.close()
        if args.tune:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
