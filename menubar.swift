// A menu-bar switch for flipoff. Native AppKit so it pulls in no dependencies
// at all -- swiftc ships with the Command Line Tools.
//
// Build:  swiftc -O menubar.swift -o flipoff-menu
// Run:    ./flipoff-menu
//
// The detector runs as a child process. That matters for permissions: macOS
// attributes Camera and Accessibility to the *responsible* process, so once you
// launch it from here the grants belong to this app rather than to your
// terminal. Being a real GUI app it can show the prompt, which is exactly what
// the LaunchAgent could not do.

import Cocoa

// Wherever this binary was built to -- the detector, log and args file all sit
// beside it, so a clone works without editing anything.
private let dir = (Bundle.main.executableURL?.resolvingSymlinksInPath()
    .deletingLastPathComponent().path) ?? FileManager.default.currentDirectoryPath
private let exe = "\(dir)/flipoff"
private let logPath = "\(dir)/flipoff.log"
private let argsPath = "\(dir)/args"

final class Controller: NSObject, NSApplicationDelegate {
    private var item: NSStatusItem!
    private var task: Process?

    /// Extra flags live in a plain text file so they can be changed without a
    /// rebuild. One argument per line, which sidesteps quoting entirely -- a
    /// recipient like `Jane Doe` has a space in it and would not survive a
    /// whitespace split.
    private var arguments: [String] {
        guard let raw = try? String(contentsOfFile: argsPath, encoding: .utf8) else {
            return ["--hand", "any"]
        }
        return raw.split(separator: "\n")
                  .map { $0.trimmingCharacters(in: .whitespaces) }
                  .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    /// Ask the system, rather than trusting our own handle. A detector that was
    /// orphaned, started from a terminal, or left over from a previous launch is
    /// still holding the camera -- a stale `Process` reference would report it
    /// as off and offer to start a second one.
    private var isRunning: Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        p.arguments = ["-f", "flipoff\\.py"]
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        do { try p.run() } catch { return task?.isRunning ?? false }
        p.waitUntilExit()
        return p.terminationStatus == 0
    }

    /// Armed state survives a relaunch, so turning it on is a one-time act
    /// rather than something to redo after every reboot.
    private var wasArmed: Bool {
        get { UserDefaults.standard.bool(forKey: "armed") }
        set { UserDefaults.standard.set(newValue, forKey: "armed") }
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.accessory)   // menu bar only, no Dock icon
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if wasArmed && !isRunning { start() }
        render()
        // The detector can die or be killed from a terminal without telling us,
        // so re-read the truth on a timer instead of only on click.
        Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.render()
        }
        // The child is not reparented on quit, so make sure it dies with us.
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil, queue: .main) { [weak self] _ in self?.stop() }
    }

    private func render() {
        let on = isRunning
        if let button = item.button {
            // Dimmed when off, full strength when armed -- readable at a glance
            // in both light and dark menu bars without shipping an icon set.
            button.attributedTitle = NSAttributedString(
                string: "🖕",
                attributes: [.foregroundColor: NSColor.labelColor.withAlphaComponent(on ? 1.0 : 0.35)])
            button.toolTip = on ? "flipoff is armed" : "flipoff is off"
        }

        let menu = NSMenu()
        let toggle = NSMenuItem(title: on ? "Armed — flipping off sends a text"
                                          : "Off",
                                action: #selector(toggle(_:)), keyEquivalent: "")
        toggle.target = self
        toggle.state = on ? .on : .off
        menu.addItem(toggle)
        menu.addItem(.separator())

        let log = NSMenuItem(title: "Open log…", action: #selector(openLog(_:)),
                             keyEquivalent: "l")
        log.target = self
        menu.addItem(log)

        let quit = NSMenuItem(title: "Quit", action: #selector(quit(_:)),
                              keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        item.menu = menu
    }

    @objc private func toggle(_ sender: Any?) {
        if isRunning {
            stop()
            wasArmed = false
        } else {
            start()
            wasArmed = isRunning
        }
        render()
    }

    private func start() {
        stop()                                  // never leave two on the camera
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = arguments
        p.currentDirectoryURL = URL(fileURLWithPath: dir)
        if let handle = FileHandle(forWritingAtPath: logPath) {
            handle.seekToEndOfFile()
            p.standardOutput = handle
            p.standardError = handle
        }
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async { self?.render() }
        }
        do {
            try p.run()
            task = p
        } catch {
            let alert = NSAlert()
            alert.messageText = "Could not start flipoff"
            alert.informativeText = "\(exe)\n\n\(error.localizedDescription)"
            alert.runModal()
        }
    }

    private func stop() {
        task?.terminate()
        task = nil
        // The launcher execs uv, which spawns python; terminating the wrapper
        // can orphan the grandchild, so sweep by name as well.
        let kill = Process()
        kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        kill.arguments = ["-f", "flipoff.py"]
        try? kill.run()
        kill.waitUntilExit()
    }

    @objc private func openLog(_ sender: Any?) {
        NSWorkspace.shared.open(URL(fileURLWithPath: logPath))
    }

    @objc private func quit(_ sender: Any?) {
        stop()
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.run()
