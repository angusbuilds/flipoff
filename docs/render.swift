// Renders the repo's images with AppKit, so the middle finger is the real Apple
// Color Emoji glyph and the bubble is drawn at native resolution rather than
// scaled from a screenshot. No dependencies: swiftc ships with the CLT.
//
//   swiftc -O docs/render.swift -o /tmp/flipoff-render && /tmp/flipoff-render
//
// Writes beside this source file, so it works from any clone.

import AppKit

// Output lands next to this file whatever machine it was cloned onto.
let DOCS = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

let BLUE = NSColor(srgbRed: 0.208, green: 0.596, blue: 0.949, alpha: 1)

/// Light and dark are the same drawing with four colours swapped, so the two
/// versions of an image can never drift apart.
struct Theme {
    let bg, ink, mute, meta: NSColor

    static let light = Theme(
        bg:   NSColor(srgbRed: 1.000, green: 1.000, blue: 1.000, alpha: 1),
        ink:  NSColor(srgbRed: 0.067, green: 0.075, blue: 0.094, alpha: 1),
        mute: NSColor(srgbRed: 0.478, green: 0.510, blue: 0.561, alpha: 1),
        meta: NSColor(srgbRed: 0.557, green: 0.557, blue: 0.576, alpha: 1))

    static let dark = Theme(
        bg:   NSColor(srgbRed: 0.051, green: 0.067, blue: 0.090, alpha: 1),
        ink:  NSColor(srgbRed: 0.941, green: 0.953, blue: 0.973, alpha: 1),
        mute: NSColor(srgbRed: 0.545, green: 0.580, blue: 0.639, alpha: 1),
        meta: NSColor(srgbRed: 0.404, green: 0.435, blue: 0.490, alpha: 1))
}

func font(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
    NSFont.systemFont(ofSize: size, weight: weight)
}

func draw(_ s: String, _ attrs: [NSAttributedString.Key: Any], at p: CGPoint) {
    NSAttributedString(string: s, attributes: attrs).draw(at: p)
}

func size(_ s: String, _ attrs: [NSAttributedString.Key: Any]) -> CGSize {
    NSAttributedString(string: s, attributes: attrs).size()
}

/// The iMessage bubble: pill body, the little tail, and "Delivered" beneath.
func bubble(text: String, origin: CGPoint, fontSize: CGFloat, theme: Theme) -> CGSize {
    let label: [NSAttributedString.Key: Any] = [
        .font: font(fontSize, .bold), .foregroundColor: NSColor.white]
    let tw = size(text, label).width
    let padX = fontSize * 0.95, padY = fontSize * 0.62
    let w = tw + padX * 2, h = fontSize * 1.25 + padY * 2
    let r = NSRect(x: origin.x, y: origin.y, width: w, height: h)

    BLUE.setFill()
    NSBezierPath(roundedRect: r, xRadius: h / 2, yRadius: h / 2).fill()

    // tail: the little hook iMessage draws off the bottom-right, rounded
    // rather than pointed -- a sharp triangle reads as a speech balloon, not
    // as a message bubble.
    let tail = NSBezierPath()
    let bx = r.maxX - h * 0.22, by = r.minY
    tail.move(to: CGPoint(x: bx, y: by + h * 0.30))
    tail.curve(to: CGPoint(x: bx + h * 0.34, y: by + h * 0.02),
               controlPoint1: CGPoint(x: bx + h * 0.20, y: by + h * 0.26),
               controlPoint2: CGPoint(x: bx + h * 0.30, y: by + h * 0.14))
    tail.curve(to: CGPoint(x: bx + h * 0.10, y: by + h * 0.02),
               controlPoint1: CGPoint(x: bx + h * 0.22, y: by - h * 0.02),
               controlPoint2: CGPoint(x: bx + h * 0.16, y: by - h * 0.01))
    tail.curve(to: CGPoint(x: bx - h * 0.10, y: by + h * 0.26),
               controlPoint1: CGPoint(x: bx + h * 0.01, y: by + h * 0.06),
               controlPoint2: CGPoint(x: bx - h * 0.06, y: by + h * 0.16))
    tail.close()
    tail.fill()

    draw(text, label, at: CGPoint(x: r.minX + padX,
                                  y: r.midY - size(text, label).height / 2))

    let delivered: [NSAttributedString.Key: Any] = [
        .font: font(fontSize * 0.52, .regular), .foregroundColor: theme.meta]
    let dw = size("Delivered", delivered).width
    draw("Delivered", delivered,
         at: CGPoint(x: r.maxX - dw, y: r.minY - fontSize * 1.15))
    return CGSize(width: w, height: h)
}

func emoji(_ glyph: String, size s: CGFloat, at p: CGPoint) {
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont(name: "AppleColorEmoji", size: s) ?? font(s)]
    draw(glyph, attrs, at: p)
}

func render(_ w: Int, _ h: Int, _ theme: Theme, _ body: () -> Void) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    theme.bg.setFill()
    NSRect(x: 0, y: 0, width: w, height: h).fill()
    body()
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

func write(_ data: Data, _ name: String) {
    try! data.write(to: DOCS.appendingPathComponent(name))
    print("wrote docs/\(name)")
}

// ── social preview, 1280×640 ────────────────────────────────────────────────
// This is the link card X and GitHub unfurl, so it is the first thing anyone
// sees. Dark, because a white card on a dark timeline reads as a blank slab,
// and because the yellow glyph and the blue bubble both glow against it.
func social(_ theme: Theme) -> Data {
    render(1280, 640, theme) {
        emoji("🖕", size: 340, at: CGPoint(x: 158, y: 152))

        draw("flipoff", [.font: font(88, .bold), .foregroundColor: theme.ink],
             at: CGPoint(x: 566, y: 392))

        draw("Flip off your webcam. It texts for you.",
             [.font: font(31, .regular), .foregroundColor: theme.mute],
             at: CGPoint(x: 571, y: 340))

        _ = bubble(text: "FUCK YOU", origin: CGPoint(x: 571, y: 168),
                   fontSize: 44, theme: theme)

        // Survives being screenshotted and reposted without the link.
        draw("github.com/angusbuilds/flipoff",
             [.font: NSFont.monospacedSystemFont(ofSize: 19, weight: .regular),
              .foregroundColor: theme.meta],
             at: CGPoint(x: 571, y: 74))
    }
}
write(social(.dark), "social.png")

// ── README hero: sized to the content, not padded out to a banner ─────────
// Two files so the README can swap them on prefers-color-scheme; a white hero
// on GitHub's dark theme is a bright slab across the top of the page.
func hero(_ theme: Theme) -> Data {
    render(896, 372, theme) {
        emoji("🖕", size: 200, at: CGPoint(x: 96, y: 96))

        draw("flipoff", [.font: font(62, .bold), .foregroundColor: theme.ink],
             at: CGPoint(x: 372, y: 238))

        draw("Flip off your webcam. It texts for you.",
             [.font: font(25, .regular), .foregroundColor: theme.mute],
             at: CGPoint(x: 376, y: 198))

        _ = bubble(text: "FUCK YOU", origin: CGPoint(x: 376, y: 86),
                   fontSize: 32, theme: theme)
    }
}
write(hero(.light), "hero.png")
write(hero(.dark), "hero-dark.png")
