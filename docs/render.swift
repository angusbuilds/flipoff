// Renders the repo's images with AppKit, so the middle finger is the real Apple
// Color Emoji glyph and the bubble is drawn at native resolution rather than
// scaled from a screenshot. No dependencies: swiftc ships with the CLT.
//
//   swiftc -O docs/render.swift -o /tmp/flipoff-render && /tmp/flipoff-render

import AppKit

let INK  = NSColor(srgbRed: 0.067, green: 0.075, blue: 0.094, alpha: 1)
let MUTE = NSColor(srgbRed: 0.478, green: 0.510, blue: 0.561, alpha: 1)
let BLUE = NSColor(srgbRed: 0.208, green: 0.596, blue: 0.949, alpha: 1)
let GREY = NSColor(srgbRed: 0.557, green: 0.557, blue: 0.576, alpha: 1)

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
func bubble(text: String, origin: CGPoint, fontSize: CGFloat) -> CGSize {
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
        .font: font(fontSize * 0.52, .regular), .foregroundColor: GREY]
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

func render(_ w: Int, _ h: Int, _ body: () -> Void) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: w, height: h).fill()
    body()
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

// ── social preview, 1280×640 ────────────────────────────────────────────────
let hero = render(1280, 640) {
    emoji("🖕", size: 300, at: CGPoint(x: 150, y: 158))

    let title: [NSAttributedString.Key: Any] = [
        .font: font(76, .bold), .foregroundColor: INK]
    draw("flipoff", title, at: CGPoint(x: 546, y: 398))

    let sub: [NSAttributedString.Key: Any] = [
        .font: font(30, .regular), .foregroundColor: MUTE]
    draw("Flip off your webcam. It texts for you.", sub,
         at: CGPoint(x: 550, y: 350))

    _ = bubble(text: "FUCK YOU", origin: CGPoint(x: 550, y: 196), fontSize: 40)
}
try! hero.write(to: URL(fileURLWithPath: "/Users/angus/flipoff/docs/social.png"))

// ── README hero: sized to the content, not padded out to a banner ─────────
let banner = render(1040, 380) {
    emoji("🖕", size: 200, at: CGPoint(x: 96, y: 96))

    let title: [NSAttributedString.Key: Any] = [
        .font: font(62, .bold), .foregroundColor: INK]
    draw("flipoff", title, at: CGPoint(x: 372, y: 238))

    let sub: [NSAttributedString.Key: Any] = [
        .font: font(25, .regular), .foregroundColor: MUTE]
    draw("Flip off your webcam. It texts for you.", sub, at: CGPoint(x: 376, y: 198))

    _ = bubble(text: "FUCK YOU", origin: CGPoint(x: 376, y: 86), fontSize: 32)
}
try! banner.write(to: URL(fileURLWithPath: "/Users/angus/flipoff/docs/hero.png"))

print("wrote docs/social.png (1280x640) and docs/hero.png (1600x480)")
