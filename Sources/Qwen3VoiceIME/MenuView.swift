import Carbon
import SwiftUI

struct MenuView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(appState.statusText)
                .font(.system(size: 12))
                .frame(maxWidth: 260, alignment: .leading)

            if !appState.lastTranscription.isEmpty {
                Divider()
                Text(appState.lastTranscription)
                    .font(.system(size: 12))
                    .frame(maxWidth: 260, alignment: .leading)
                    .lineLimit(6)
                    .textSelection(.enabled)
            }

            if let progress = appState.downloadProgress {
                ProgressView(value: progress)
                    .frame(width: 240)
            }

            if let error = appState.lastError {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.system(size: 11))
                    .frame(maxWidth: 260, alignment: .leading)
            }

            Divider()

            recordButton

            SettingsLink {
                Text("打开设置")
            }

            if !TextInsertion.isTrustedForInput() {
                Button("请求辅助功能权限") {
                    TextInsertion.requestAccessibilityPermission()
                }
            }

            Divider()

            Button("退出") {
                appState.quit()
            }
            .keyboardShortcut("q")
        }
        .padding(12)
        .frame(width: 500)
    }
}

private extension MenuView {
    struct RecordButtonShortcut {
        let key: SwiftUI.KeyEquivalent
        let modifiers: SwiftUI.EventModifiers
    }

    var recordButtonBaseTitle: String {
        appState.isRecording ? "停止并转写" : "开始识别"
    }

    @ViewBuilder
    var recordButton: some View {
        let label = Text(recordButtonBaseTitle)
        if let shortcut = recordButtonShortcut {
            Button {
                Task { @MainActor in
                    if appState.isRecording {
                        await appState.stopAndTranscribe()
                    } else {
                        await appState.startRecording()
                    }
                }
            } label: {
                label
            }
            .disabled(appState.isBusy)
            .keyboardShortcut(shortcut.key, modifiers: shortcut.modifiers)
        } else {
            Button {
                Task { @MainActor in
                    if appState.isRecording {
                        await appState.stopAndTranscribe()
                    } else {
                        await appState.startRecording()
                    }
                }
            } label: {
                label
            }
            .disabled(appState.isBusy)
        }
    }

    var recordButtonShortcut: RecordButtonShortcut? {
        let hotKey = appState.settings.hotKey
        if !hotKey.isEnabled { return nil }
        guard let key = keyEquivalent(for: hotKey.keyCode) else { return nil }
        let modifiers = eventModifiers(for: hotKey.modifiers)
        return RecordButtonShortcut(key: key, modifiers: modifiers)
    }

    func keyEquivalent(for keyCode: UInt32) -> SwiftUI.KeyEquivalent? {
        let name = KeyCodeFormatter.displayName(keyCode: keyCode)
        if name.count == 1, let c = name.lowercased().first {
            return SwiftUI.KeyEquivalent(c)
        }
        switch name {
        case "Space":
            return .space
        case "↩︎":
            return .return
        case "⌫":
            return .delete
        case "⎋":
            return .escape
        default:
            return nil
        }
    }

    func eventModifiers(for mask: UInt32) -> SwiftUI.EventModifiers {
        var mods: SwiftUI.EventModifiers = []
        if (mask & UInt32(cmdKey)) != 0 { mods.insert(.command) }
        if (mask & UInt32(optionKey)) != 0 { mods.insert(.option) }
        if (mask & UInt32(controlKey)) != 0 { mods.insert(.control) }
        if (mask & UInt32(shiftKey)) != 0 { mods.insert(.shift) }
        return mods
    }
}
