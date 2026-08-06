#!/usr/bin/env python3
"""Draw docs/tune.svg -- what `--tune` puts on screen, from real recorded hands.

Three frames out of fixtures.json, scored by the same rules flipoff.py uses and
drawn the way the tune window draws them: skeleton, middle-finger chain picked
out, and the score bar with the threshold tick. Nothing here is illustrated by
hand, so the bars are the scores the classifier actually produces.

    ./docs/tuneshot.py
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WRIST = 0
INDEX, MIDDLE, RING, PINKY = (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)
FINGERS, FOLDERS = (INDEX, MIDDLE, RING, PINKY), (INDEX, RING, PINKY)
CONNECTIONS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
               (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
               (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
               (17, 18), (18, 19), (19, 20))

# Which frames tell the story: a clean gesture, the nearest miss in the whole
# fixture set, and a hand doing nothing like it. Picked by eye off a contact
# sheet of all 24 -- several frames are hands aimed down the lens, where the
# flat projection collapses into scribble even though the classifier, which
# reads all three axes, scores them fine.
PANELS = [("gesture", 2, "a real flip-off"),
          ("not_gesture", 6, "the nearest miss on file"),
          ("not_gesture", 9, "an ordinary open hand")]

W, H = 1000, 414
PANEL_W, PANEL_H, PANEL_Y = 300, 236, 84
GAP = (W - 52 - 3 * PANEL_W) / 2
THRESHOLD = 0.5


def consts():
    src = (ROOT / "flipoff.py").read_text()

    def pair(name):
        m = re.search(rf"^{name} = \(([\d.]+), ([\d.]+)\)", src, re.M)
        return (float(m[1]), float(m[2]))

    def one(name):
        return float(re.search(rf"^{name} = ([\d.]+)", src, re.M)[1])

    return (pair("MIDDLE_EXTENDED"), pair("OTHER_FOLDED"), pair("RING_FOLDED"),
            one("MIN_MIDDLE_REACH"), one("REACH_MARGIN"))


MID_EXT, OTHER_FOLDED, RING_FOLDED, MIN_REACH, REACH_MARGIN = consts()
BAND = {INDEX: OTHER_FOLDED, RING: RING_FOLDED, PINKY: OTHER_FOLDED}


def extension(p, f):
    j = [p[k] for k in f]
    path = sum(math.dist(j[k], j[k + 1]) for k in range(3))
    return math.dist(j[0], j[3]) / path if path > 1e-9 else 0.0


def sat(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def score(p):
    """Same gates and the same min() as flipoff.score_hand."""
    ext = {f: extension(p, f) for f in FINGERS}
    base = math.dist(p[MIDDLE[0]], p[WRIST])
    if base < 1e-9 or math.dist(p[MIDDLE[3]], p[WRIST]) / base < MIN_REACH:
        return 0.0, ext
    reach = math.dist(p[MIDDLE[3]], p[WRIST])
    for f in FOLDERS:
        if reach < REACH_MARGIN * math.dist(p[f[3]], p[WRIST]):
            return 0.0, ext
    parts = [sat(ext[MIDDLE], *MID_EXT)] + [sat(ext[f], *BAND[f]) for f in FOLDERS]
    return min(parts), ext


def project(pts, box):
    """Fit the hand into box=(x, y, w, h), fingers pointing up.

    Only x and y are drawn. The z axis is what makes the measurement work but
    it is also what a flat picture cannot show, which is the whole reason the
    classifier scores world landmarks instead of the 2D ones.
    """
    # These frames were recorded at whatever angle the hand happened to be held,
    # which makes three of them side by side unreadable. Rotate each one so the
    # wrist-to-middle-knuckle axis stands straight up: same hand, same score,
    # just presented from a common angle.
    z = [complex(p[0], p[1]) for p in pts]
    v = z[MIDDLE[0]] - z[WRIST]
    turn = 1j * v.conjugate() / abs(v) if abs(v) > 1e-9 else 1 + 0j
    z = [(w - z[WRIST]) * turn for w in z]

    xs, ys = [w.real for w in z], [w.imag for w in z]
    bx, by, bw, bh = box
    spread = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scale = min(bw, bh) * 0.86 / spread
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    # SVG y grows downward, so subtract to send +y up the page.
    return [(bx + bw / 2 + (x - cx) * scale, by + bh / 2 - (y - cy) * scale)
            for x, y in zip(xs, ys)]


def main():
    fixtures = json.loads((ROOT / "fixtures.json").read_text())
    out = []
    add = out.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="-apple-system, BlinkMacSystemFont, '
        f"'Segoe UI', Helvetica, Arial, sans-serif\">")
    add("""  <style>
    .bg    { fill: #ffffff }
    .cam   { fill: #f6f8fc; stroke: #e3e8f0; stroke-width: 1.5 }
    .title { fill: #111318; font-size: 21px; font-weight: 700 }
    .sub   { fill: #7a828f; font-size: 13.5px }
    .cap   { fill: #57606a; font-size: 13px; font-weight: 600 }
    .bone  { stroke: #b9c2d0; stroke-width: 2.4; stroke-linecap: round; fill: none }
    .joint { fill: #b9c2d0 }
    .track { fill: #e6eaf2 }
    .mono  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: 12px; fill: #7a828f }
    .tickl { stroke: #9aa3b2; stroke-width: 1.5 }
    .hot   { stroke: #e0245e; stroke-width: 3.2; stroke-linecap: round; fill: none }
    .hotj  { fill: #e0245e }
    .fire  { fill: #1a9c53 }
    .cold  { fill: #9aa3b2 }
    @media (prefers-color-scheme: dark) {
      .bg    { fill: #0b0d12 }
      .cam   { fill: #10151f; stroke: #222a38 }
      .title { fill: #e8ebf5 }
      .sub   { fill: #8b949e }
      .cap   { fill: #aab3c0 }
      .bone  { stroke: #3d4657 }
      .joint { fill: #3d4657 }
      .track { fill: #1a2130 }
      .mono  { fill: #79828f }
      .tickl { stroke: #5b6472 }
      .hot   { stroke: #ff5c8a }
      .hotj  { fill: #ff5c8a }
      .fire  { fill: #2ea043 }
      .cold  { fill: #5b6472 }
    }
  </style>""")
    add(f'  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')
    add('  <text x="26" y="38" class="title">What <tspan class="mono" '
        'font-size="19">--tune</tspan> shows you</text>')
    add('  <text x="26" y="61" class="sub">Three recorded frames, scored by the same '
        'rules that decide whether to send. Green past the tick means it would fire.'
        '</text>')

    hot_bones = {(9, 10), (10, 11), (11, 12), (5, 9)}

    for i, (key, idx, caption) in enumerate(PANELS):
        frame = fixtures[key][idx]
        s, ext = score(frame)
        px = 26 + i * (PANEL_W + GAP)

        add(f'  <rect class="cam" x="{px}" y="{PANEL_Y}" width="{PANEL_W}" '
            f'height="{PANEL_H}" rx="12"/>')

        pts = project(frame, (px, PANEL_Y, PANEL_W, PANEL_H))
        for a, b in CONNECTIONS:
            cls = "hot" if (a, b) in hot_bones else "bone"
            add(f'  <line class="{cls}" x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" '
                f'x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}"/>')
        for n, (x, yy) in enumerate(pts):
            hot = n in MIDDLE
            add(f'  <circle class="{"hotj" if hot else "joint"}" cx="{x:.1f}" '
                f'cy="{yy:.1f}" r="{4.2 if hot else 3.2}"/>')

        # score readout and bar, laid out like the tune window's overlay
        by = PANEL_Y + PANEL_H + 30
        add(f'  <text x="{px}" y="{by-6}" class="mono">middle {ext[MIDDLE]:.2f}'
            f'   index {ext[INDEX]:.2f}   ring {ext[RING]:.2f}   '
            f'pinky {ext[PINKY]:.2f}</text>')
        add(f'  <rect class="track" x="{px}" y="{by+4}" width="{PANEL_W}" '
            f'height="14" rx="7"/>')
        if s > 0:
            add(f'  <rect class="{"fire" if s >= THRESHOLD else "cold"}" x="{px}" '
                f'y="{by+4}" width="{max(PANEL_W*s, 14):.1f}" height="14" rx="7"/>')
        tx = px + PANEL_W * THRESHOLD
        add(f'  <line class="tickl" x1="{tx:.1f}" y1="{by}" x2="{tx:.1f}" '
            f'y2="{by+22}"/>')
        add(f'  <text x="{px}" y="{by+44}" class="cap">{s:.2f}  {caption}</text>')

    add("</svg>")
    (ROOT / "docs" / "tune.svg").write_text("\n".join(out) + "\n")
    print("wrote docs/tune.svg")
    for key, idx, cap in PANELS:
        print(f"  {score(fixtures[key][idx])[0]:.2f}  {cap}")


if __name__ == "__main__":
    main()
