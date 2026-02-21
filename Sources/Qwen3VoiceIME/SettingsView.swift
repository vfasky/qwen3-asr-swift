import Carbon
import AppKit
import SwiftUI
import Qwen3Common

struct SettingsView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            hotKeySection

            Divider()

            modelSection

            Divider()

            permissionSection

            Spacer(minLength: 0)
        }
        .padding(24)
        .frame(width: 520)
        .background(SettingsWindowActivator().frame(width: 0, height: 0))
    }
}

private extension SettingsView {
    var hotKeySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("热键")
                .font(.headline)

            HStack(spacing: 16) {
                Toggle("启用热键", isOn: Binding(
                    get: { appState.settings.hotKey.isEnabled },
                    set: { appState.settings.hotKey.isEnabled = $0 }
                ))

                Spacer()

                Text("当前热键")
                    .foregroundStyle(.secondary)
                Text(HotKeyFormatting.displayString(for: appState.settings.hotKey))
                    .font(.system(.body, design: .monospaced))
            }

            HotKeyRecorderButton(hotKey: $appState.settings.hotKey)
        }
    }

    var modelSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("模型")
                .font(.headline)

            let modelId = appState.settings.modelSize.defaultModelId
            let cachePath = (try? ModelDownloader.getCacheDirectory(for: modelId).path) ?? "-"

            HStack(alignment: .center, spacing: 16) {
                Text("模型大小")
                    .frame(width: 80, alignment: .trailing)
                    .foregroundStyle(.secondary)
                Picker("", selection: $appState.settings.modelSize) {
                    ForEach(ASRModelChoice.allCases, id: \.self) { choice in
                        Text(choice.displayName).tag(choice)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 200, alignment: .leading)
            }

            HStack(alignment: .center, spacing: 16) {
                Text("下载源")
                    .frame(width: 80, alignment: .trailing)
                    .foregroundStyle(.secondary)
                Picker("", selection: $appState.settings.downloadSource) {
                    Text("ModelScope").tag(ModelDownloadSource.modelscope)
                    Text("HuggingFace").tag(ModelDownloadSource.huggingface)
                }
                .labelsHidden()
                .frame(maxWidth: 200, alignment: .leading)
            }

            HStack(alignment: .top, spacing: 16) {
                Text("模型 ID")
                    .frame(width: 80, alignment: .trailing)
                    .foregroundStyle(.secondary)
                Text(modelId)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(alignment: .top, spacing: 16) {
                Text("缓存目录")
                    .frame(width: 80, alignment: .trailing)
                    .foregroundStyle(.secondary)
                Text(cachePath)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 16) {
                Spacer()
                    .frame(width: 80)
                Button("下载/准备模型") {
                    Task { @MainActor in
                        await appState.prepareModel()
                    }
                }
                .disabled(appState.isBusy)

                if let progress = appState.downloadProgress {
                    ProgressView(value: progress)
                        .frame(width: 160)
                }
            }
        }
    }

    var permissionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("权限")
                .font(.headline)

            if !TextInsertion.isTrustedForInput() {
                Button("请求辅助功能权限") {
                    TextInsertion.requestAccessibilityPermission()
                }
                Text("用于将转写结果输入到前台应用（模拟粘贴）。")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            } else {
                Text("辅助功能权限已授权")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct HotKeyRecorderButton: View {
    @Binding var hotKey: HotKey
    @State private var isCapturing = false
    @State private var monitor: Any? = nil

    var body: some View {
        HStack {
            Button(isCapturing ? "按下新热键…" : "录制热键") {
                if isCapturing {
                    stopCapture()
                } else {
                    startCapture()
                }
            }
            Button("恢复默认") {
                hotKey = .default
            }
        }
        .onDisappear {
            stopCapture()
        }
    }

    private func startCapture() {
        isCapturing = true
        monitor = NSEvent.addGlobalMonitorForEvents(matching: [.keyDown]) { event in
            let modifiers = carbonModifiers(from: event.modifierFlags)
            hotKey = HotKey(keyCode: UInt32(event.keyCode), modifiers: modifiers, isEnabled: true)
            stopCapture()
        }
    }

    private func stopCapture() {
        isCapturing = false
        if let monitor {
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }
    }

    private func carbonModifiers(from flags: NSEvent.ModifierFlags) -> UInt32 {
        var mods: UInt32 = 0
        if flags.contains(.command) { mods |= UInt32(cmdKey) }
        if flags.contains(.option) { mods |= UInt32(optionKey) }
        if flags.contains(.control) { mods |= UInt32(controlKey) }
        if flags.contains(.shift) { mods |= UInt32(shiftKey) }
        return mods
    }
}

private struct SettingsWindowActivator: NSViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSView {
        NSView()
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        guard !context.coordinator.didActivate else { return }
        guard let window = nsView.window else { return }
        context.coordinator.didActivate = true
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    final class Coordinator {
        var didActivate = false
    }
}
