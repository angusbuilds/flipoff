#!/usr/bin/env python3
"""Draw docs/detection.svg from fixtures.json and the bands in flipoff.py.

The figure used to be a hand-built image, which is how it ended up advertising
thresholds the code had already moved off. This reads both the data and the
bands from source, so the picture cannot disagree with what ships. Stdlib only,
and SVG rather than PNG so it stays sharp and can follow the reader's theme.

    ./docs/chart.py
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

W, H = 1000, 540
PAD_L, PAD_R, PAD_T, PAD_B = 74, 28, 108, 80
LO, HI = 0.28, 1.06          # y range, in extension units


def bands():
    """(low, high) per finger, lifted straight out of flipoff.py."""
    src = (ROOT / "flipoff.py").read_text()

    def pair(name):
        m = re.search(rf"^{name} = \(([\d.]+), ([\d.]+)\)", src, re.M)
        if not m:
            raise SystemExit(f"{name} not found in flipoff.py -- did it get renamed?")
        return float(m[1]), float(m[2])

    mid, other, ring = pair("MIDDLE_EXTENDED"), pair("OTHER_FOLDED"), pair("RING_FOLDED")
    return {"index": other, "middle": mid, "ring": ring, "pinky": other}


def extension(frame, finger):
    """Straight-line knuckle-to-tip over the bone path. Mirrors flipoff.py."""
    j = [frame[k] for k in finger]
    dist = lambda a, b: math.dist(a, b)
    path = sum(dist(j[k], j[k + 1]) for k in range(3))
    return dist(j[0], j[3]) / path if path > 1e-9 else 0.0


def y(v):
    return PAD_T + (HI - v) / (HI - LO) * (H - PAD_T - PAD_B)


def beeswarm(values, cx, half=26.0):
    """Deterministic swarm: bin by height, fan each bin out around cx.

    No RNG, so regenerating the figure produces a byte-identical file and a
    diff on it always means the data or the bands actually moved.
    """
    bins = {}
    for v in values:
        bins.setdefault(round(y(v) / 7), []).append(v)
    out = []
    for _, group in sorted(bins.items()):
        n = len(group)
        for i, v in enumerate(sorted(group)):
            offset = 0.0 if n == 1 else (i / (n - 1) - 0.5) * 2 * min(half, 5.0 * n)
            out.append((cx + offset, y(v)))
    return out


def main():
    fixtures = json.loads((ROOT / "fixtures.json").read_text())
    band = bands()
    cols = list(FINGERS)
    span = (W - PAD_L - PAD_R) / len(cols)

    p = []
    add = p.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="-apple-system, BlinkMacSystemFont, '
        f"'Segoe UI', Helvetica, Arial, sans-serif\">")
    add("""  <style>
    .bg    { fill: #ffffff }
    .title { fill: #111318; font-size: 21px; font-weight: 700 }
    .sub   { fill: #7a828f; font-size: 13.5px }
    .axis  { fill: #7a828f; font-size: 12.5px }
    .tick  { fill: #aab2c0; font-size: 11.5px }
    .grid  { stroke: #eef1f6; stroke-width: 1 }
    .frame { stroke: #e3e8f0; stroke-width: 1; fill: none }
    .band  { fill: #eaeefa }
    .note  { fill: #8d95a3; font-size: 11.5px }
    .yes   { fill: #e0245e }
    .no    { fill: #3b9dff }
    .lgnd  { fill: #57606a; font-size: 13px }
    .edge  { fill: #9aa3b2; font-size: 10.5px; font-weight: 600 }
    @media (prefers-color-scheme: dark) {
      .bg    { fill: #0b0d12 }
      .title { fill: #e8ebf5 }
      .sub   { fill: #8b949e }
      .axis  { fill: #8b949e }
      .tick  { fill: #6b7480 }
      .grid  { stroke: #171c25 }
      .frame { stroke: #222834 }
      .band  { fill: #1b2231 }
      .note  { fill: #79828f }
      .lgnd  { fill: #9aa4b1 }
      .edge  { fill: #6e7787 }
    }
  </style>""")
    add(f'  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')

    add('  <text x="26" y="38" class="title">What a real flip-off looks like to '
        'the classifier</text>')
    add('  <text x="26" y="59" class="sub">Every point is one frame recorded off an '
        'actual webcam. The shaded band is that finger&#8217;s scoring ramp, read '
        'straight from flipoff.py.</text>')
    add('  <text x="26" y="78" class="sub">The middle finger has to clear the top of '
        'its band; the other three have to fall below the bottom of theirs.</text>')

    # legend
    add(f'  <circle class="yes" cx="{W-292}" cy="34" r="5.5"/>'
        f'<text x="{W-278}" y="39" class="lgnd">real flip-off</text>')
    add(f'  <circle class="no" cx="{W-150}" cy="34" r="5.5"/>'
        f'<text x="{W-136}" y="39" class="lgnd">ordinary hand</text>')

    # gridlines + y ticks
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        add(f'  <line class="grid" x1="{PAD_L}" y1="{y(t):.1f}" x2="{W-PAD_R}" '
            f'y2="{y(t):.1f}"/>')
        add(f'  <text x="{PAD_L-11}" y="{y(t)+4:.1f}" class="tick" '
            f'text-anchor="end">{t:.1f}</text>')
    add(f'  <text transform="translate(22,{(PAD_T+H-PAD_B)/2}) rotate(-90)" '
        f'class="axis" text-anchor="middle">extension  (1.0 = perfectly straight)'
        f'</text>')

    # bands, points, labels -- one column per finger
    for i, name in enumerate(cols):
        cx = PAD_L + span * (i + 0.5)
        lo, hi = sorted(band[name])
        add(f'  <rect class="band" x="{cx-span*0.42:.1f}" y="{y(hi):.1f}" '
            f'width="{span*0.84:.1f}" height="{y(lo)-y(hi):.1f}" rx="3"/>')

        for key, cls in (("not_gesture", "no"), ("gesture", "yes")):
            vals = [extension(f, FINGERS[name]) for f in fixtures[key]]
            shift = -span * 0.19 if key == "gesture" else span * 0.19
            for px, py in beeswarm(vals, cx + shift):
                add(f'  <circle class="{cls}" cx="{px:.1f}" cy="{py:.1f}" r="4"/>')

        # The band's own numbers, on its edges. Cheaper to read than a legend,
        # and it makes a moved threshold visible in the picture rather than only
        # in the source.
        edge = cx - span * 0.42 + 6
        add(f'  <text x="{edge:.1f}" y="{y(hi)+14:.1f}" class="edge">{hi:.2f}</text>')
        add(f'  <text x="{edge:.1f}" y="{y(lo)-6:.1f}" class="edge">{lo:.2f}</text>')

        want = "extended" if name == "middle" else "folded"
        add(f'  <text x="{cx:.1f}" y="{H-PAD_B+26}" class="axis" '
            f'text-anchor="middle">{name}</text>')
        add(f'  <text x="{cx:.1f}" y="{H-PAD_B+44}" class="note" '
            f'text-anchor="middle">must be {want}</text>')

    add(f'  <rect class="frame" x="{PAD_L}" y="{PAD_T-14}" width="{W-PAD_L-PAD_R}" '
        f'height="{H-PAD_T-PAD_B+14}" rx="4"/>')
    add("</svg>")

    out = ROOT / "docs" / "detection.svg"
    out.write_text("\n".join(p) + "\n")

    counts = {k: len(v) for k, v in fixtures.items()}
    print(f"wrote docs/detection.svg  "
          f"({counts['gesture']} gesture / {counts['not_gesture']} ordinary frames)")
    for name in cols:
        lo, hi = sorted(band[name])
        print(f"  {name:<7} band {lo:.2f}-{hi:.2f}")


if __name__ == "__main__":
    main()
