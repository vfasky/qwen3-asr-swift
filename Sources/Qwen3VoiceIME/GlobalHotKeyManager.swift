import Carbon
import Foundation

final class GlobalHotKeyManager {
    var onHotKey: (() -> Void)? = nil

    private var hotKeyRef: EventHotKeyRef? = nil
    private var handlerRef: EventHandlerRef? = nil

    func register(hotKey: HotKey) {
        unregister()
        guard hotKey.isEnabled else { return }

        let hotKeyID = EventHotKeyID(signature: OSType(0x51574E33), id: 1)
        let pointer = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())

        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(
            GetApplicationEventTarget(),
            { _, event, userData in
                guard let userData else { return noErr }
                let manager = Unmanaged<GlobalHotKeyManager>.fromOpaque(userData).takeUnretainedValue()
                if let event {
                    var hotKeyID = EventHotKeyID()
                    let status = GetEventParameter(
                        event,
                        EventParamName(kEventParamDirectObject),
                        EventParamType(typeEventHotKeyID),
                        nil,
                        MemoryLayout<EventHotKeyID>.size,
                        nil,
                        &hotKeyID
                    )
                    if status == noErr, hotKeyID.signature == OSType(0x51574E33) {
                        manager.onHotKey?()
                    }
                }
                return noErr
            },
            1,
            &eventType,
            pointer,
            &handlerRef
        )

        RegisterEventHotKey(hotKey.keyCode, hotKey.modifiers, hotKeyID, GetApplicationEventTarget(), 0, &hotKeyRef)
    }

    func unregister() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            self.hotKeyRef = nil
        }
        if let handlerRef {
            RemoveEventHandler(handlerRef)
            self.handlerRef = nil
        }
    }

    deinit {
        unregister()
    }
}

enum HotKeyFormatting {
    static func displayString(for hotKey: HotKey) -> String {
        if !hotKey.isEnabled { return "未启用" }

        var parts: [String] = []
        if (hotKey.modifiers & UInt32(cmdKey)) != 0 { parts.append("⌘") }
        if (hotKey.modifiers & UInt32(optionKey)) != 0 { parts.append("⌥") }
        if (hotKey.modifiers & UInt32(controlKey)) != 0 { parts.append("⌃") }
        if (hotKey.modifiers & UInt32(shiftKey)) != 0 { parts.append("⇧") }
        parts.append(KeyCodeFormatter.displayName(keyCode: hotKey.keyCode))
        return parts.joined()
    }
}

enum KeyCodeFormatter {
    static func displayName(keyCode: UInt32) -> String {
        switch keyCode {
        case 0: return "A"
        case 1: return "S"
        case 2: return "D"
        case 3: return "F"
        case 4: return "H"
        case 5: return "G"
        case 6: return "Z"
        case 7: return "X"
        case 8: return "C"
        case 9: return "V"
        case 11: return "B"
        case 12: return "Q"
        case 13: return "W"
        case 14: return "E"
        case 15: return "R"
        case 16: return "Y"
        case 17: return "T"
        case 18: return "1"
        case 19: return "2"
        case 20: return "3"
        case 21: return "4"
        case 22: return "6"
        case 23: return "5"
        case 24: return "="
        case 25: return "9"
        case 26: return "7"
        case 27: return "-"
        case 28: return "8"
        case 29: return "0"
        case 30: return "]"
        case 31: return "O"
        case 32: return "U"
        case 33: return "["
        case 34: return "I"
        case 35: return "P"
        case 36: return "↩︎"
        case 37: return "L"
        case 38: return "J"
        case 39: return "'"
        case 40: return "K"
        case 41: return ";"
        case 42: return "\\"
        case 43: return ","
        case 44: return "/"
        case 45: return "N"
        case 46: return "M"
        case 47: return "."
        case 49: return "Space"
        case 51: return "⌫"
        case 53: return "⎋"
        default: return "Key(\(keyCode))"
        }
    }
}
