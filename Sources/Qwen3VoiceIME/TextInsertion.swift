import AppKit
import ApplicationServices
import Foundation

enum TextInsertion {
    enum Result {
        case success
        case failed(String)
    }

    // 防抖状态记录，防止极短时间内的重复触发
    private static var lastInsertedText: String = ""
    private static var lastInsertTime: Date = .distantPast

    /// 同步插入（仅尝试辅助功能直接写入，不含剪贴板降级）
    /// 适用于原生 App
    static func insert(text: String, targetPID: pid_t? = nil) -> Result {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else {
            return .failed("文本为空")
        }

        // 防抖拦截
        if cleanText == lastInsertedText && abs(lastInsertTime.timeIntervalSinceNow) < 0.3 {
            return .failed("防抖：忽略极短时间内的重复调用")
        }
        lastInsertedText = cleanText
        lastInsertTime = Date()

        // 尝试辅助功能直接写入
        return insertUsingAccessibility(text: cleanText, targetPID: targetPID)
    }

    /// 异步粘贴插入（写剪贴板 → 激活目标 → 等待 → Cmd+V）
    /// 适用于 Electron 等不支持 AX 写入的应用
    @MainActor
    static func pasteInsert(text: String, targetPID: pid_t, preferMenuPaste: Bool = false) async -> Result {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else {
            return .failed("文本为空")
        }

        // 防抖拦截
        if cleanText == lastInsertedText && abs(lastInsertTime.timeIntervalSinceNow) < 0.3 {
            return .failed("防抖：忽略极短时间内的重复调用")
        }
        lastInsertedText = cleanText
        lastInsertTime = Date()

        // 1. 写入剪贴板（已在主线程）
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        guard pasteboard.setString(cleanText, forType: .string) else {
            return .failed("系统剪贴板写入失败")
        }

        // 验证剪贴板确实写入成功
        guard let verify = pasteboard.string(forType: .string), verify == cleanText else {
            return .failed("剪贴板写入验证失败")
        }

        // 2. 激活目标应用
        NSApp.hide(nil)

        let appElement = AXUIElementCreateApplication(targetPID)
        AXUIElementSetAttributeValue(appElement, kAXFrontmostAttribute as CFString, kCFBooleanTrue)
        if let app = NSRunningApplication(processIdentifier: targetPID) {
            app.activate(options: [.activateAllWindows])
        }

        // 3. 等待目标应用完全获得焦点
        try? await Task.sleep(nanoseconds: 250_000_000) // 250ms

        // 4. 再次确认目标应用已在前台
        if let app = NSRunningApplication(processIdentifier: targetPID), !app.isActive {
            app.activate(options: [.activateAllWindows])
            try? await Task.sleep(nanoseconds: 150_000_000) // 再等 150ms
        }

        // 5. 执行粘贴
        if preferMenuPaste {
            if performPasteMenuAction(targetPID: targetPID) {
                return .success
            }
            // 菜单粘贴失败，降级到按键
        }

        sendPasteKeystroke()

        // 6. 等待粘贴完成
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

        return .success
    }

    // MARK: - Public Utilities

    static func isTrustedForInput() -> Bool {
        AXIsProcessTrusted()
    }

    static func requestAccessibilityPermission() {
        let options: NSDictionary = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as NSString: true]
        _ = AXIsProcessTrustedWithOptions(options)
    }

    // MARK: - 模拟物理级别的 Cmd + V
    private static func sendPasteKeystroke() {
        let source = CGEventSource(stateID: .hidSystemState)
        let vKeyCode: CGKeyCode = 0x09 // V 键

        guard let vDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true),
              let vUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false) else { return }

        vDown.flags = .maskCommand
        vUp.flags = .maskCommand

        vDown.post(tap: .cghidEventTap)
        // 按键之间加一点间隔，某些 Electron 应用需要
        usleep(10_000) // 10ms
        vUp.post(tap: .cghidEventTap)
    }

    // MARK: - 模拟菜单点击
    private static func performPasteMenuAction(targetPID: pid_t) -> Bool {
        let appElement = AXUIElementCreateApplication(targetPID)
        AXUIElementSetAttributeValue(appElement, kAXFrontmostAttribute as CFString, kCFBooleanTrue)

        var menuBarValue: AnyObject?
        let menuResult = AXUIElementCopyAttributeValue(
            appElement,
            kAXMenuBarAttribute as CFString,
            &menuBarValue
        )
        guard menuResult == .success, let menuBarValue else { return false }
        let menuBar = unsafeBitCast(menuBarValue, to: AXUIElement.self)
        let editTitles: Set<String> = ["Edit", "编辑"]
        let pasteTitles: Set<String> = ["Paste", "粘贴"]
        if let editMenu = findMenuItem(in: menuBar, titles: editTitles, depth: 0),
           let pasteItem = findMenuItem(in: editMenu, titles: pasteTitles, depth: 0) {
            return AXUIElementPerformAction(pasteItem, kAXPressAction as CFString) == .success
        }
        guard let item = findMenuItem(in: menuBar, titles: pasteTitles, depth: 0) else { return false }
        return AXUIElementPerformAction(item, kAXPressAction as CFString) == .success
    }

    private static func findMenuItem(in element: AXUIElement, titles: Set<String>, depth: Int) -> AXUIElement? {
        if depth > 10 { return nil }
        var titleValue: AnyObject?
        if AXUIElementCopyAttributeValue(
            element,
            kAXTitleAttribute as CFString,
            &titleValue
        ) == .success, let title = titleValue as? String, titles.contains(title) {
            return element
        }
        var childrenValue: AnyObject?
        if AXUIElementCopyAttributeValue(
            element,
            kAXChildrenAttribute as CFString,
            &childrenValue
        ) == .success, let children = childrenValue as? [AnyObject] {
            for child in children {
                let childElement = unsafeBitCast(child, to: AXUIElement.self)
                if let found = findMenuItem(in: childElement, titles: titles, depth: depth + 1) {
                    return found
                }
            }
        }
        return nil
    }

    // MARK: - 辅助功能直接写入
    private static func insertUsingAccessibility(text: String, targetPID: pid_t?) -> Result {
        guard AXIsProcessTrusted() else { return .failed("未授予辅助功能权限（\(appIdentity())）") }
        if let targetPID {
            if let element = focusedElementFromApplication(pid: targetPID) {
                return insertUsingAccessibility(text: text, focusedElement: element)
            }
        }
        if let element = focusedElement(from: AXUIElementCreateSystemWide()) {
            return insertUsingAccessibility(text: text, focusedElement: element)
        }
        return .failed("无法获取焦点控件")
    }

    private static func focusedElement(from root: AXUIElement) -> AXUIElement? {
        var focused: AnyObject?
        let focusedResult = AXUIElementCopyAttributeValue(
            root,
            kAXFocusedUIElementAttribute as CFString,
            &focused
        )
        guard focusedResult == .success, let focusedElement = focused else { return nil }
        return unsafeBitCast(focusedElement, to: AXUIElement.self)
    }

    private static func focusedElementFromApplication(pid: pid_t) -> AXUIElement? {
        let appElement = AXUIElementCreateApplication(pid)
        AXUIElementSetAttributeValue(appElement, kAXFrontmostAttribute as CFString, kCFBooleanTrue)
        return focusedElement(from: appElement)
    }

    private static func insertUsingAccessibility(text: String, focusedElement: AXUIElement) -> Result {
        let element = focusedElement

        var subroleValue: AnyObject?
        if AXUIElementCopyAttributeValue(
            element,
            kAXSubroleAttribute as CFString,
            &subroleValue
        ) == .success, let subrole = subroleValue as? String {
            if subrole == (kAXSecureTextFieldSubrole as String) {
                return .failed("安全输入控件禁止写入")
            }
        }

        var selectedTextSettable = DarwinBoolean(false)
        if AXUIElementIsAttributeSettable(
            element,
            kAXSelectedTextAttribute as CFString,
            &selectedTextSettable
        ) == .success, selectedTextSettable.boolValue {
            if AXUIElementSetAttributeValue(
                element,
                kAXSelectedTextAttribute as CFString,
                text as CFTypeRef
            ) == .success {
                return .success
            }
        }

        var valueSettable = DarwinBoolean(false)
        if AXUIElementIsAttributeSettable(
            element,
            kAXValueAttribute as CFString,
            &valueSettable
        ) != .success || !valueSettable.boolValue {
            return .failed("控件不支持写入")
        }

        var currentValue: AnyObject?
        let valueResult = AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &currentValue
        )
        guard valueResult == .success, let current = currentValue as? String else {
            return .failed("无法读取控件内容")
        }

        let currentNSString = current as NSString
        let totalLength = currentNSString.length
        var range = CFRange(location: totalLength, length: 0)

        var rangeValue: AnyObject?
        if AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &rangeValue
        ) == .success, let rangeValue {
            let axValue = rangeValue as! AXValue
            if AXValueGetType(axValue) == .cfRange {
                var cfRange = CFRange()
                AXValueGetValue(axValue, .cfRange, &cfRange)
                range = cfRange
            }
        }

        let safeLocation = max(0, min(range.location, totalLength))
        let safeLength = max(0, min(range.length, totalLength - safeLocation))
        let prefix = currentNSString.substring(to: safeLocation)
        let suffix = currentNSString.substring(from: safeLocation + safeLength)
        let newValue = prefix + text + suffix

        if AXUIElementSetAttributeValue(
            element,
            kAXValueAttribute as CFString,
            newValue as CFTypeRef
        ) == .success {
            return .success
        }
        return .failed("写入失败")
    }

    private static func appIdentity() -> String {
        let bundleId = Bundle.main.bundleIdentifier ?? "未知BundleID"
        let bundlePath = Bundle.main.bundlePath
        let processName = ProcessInfo.processInfo.processName
        return "\(processName) | \(bundleId) | \(bundlePath)"
    }
}
