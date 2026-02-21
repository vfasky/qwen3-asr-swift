import Carbon
import AppKit
import SwiftUI
import Qwen3Common

struct SettingsView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        TabView {
            generalTab
                .tabItem {
                    Label("常规", systemImage: "gearshape")
                }
                
            hotwordsTab
                .tabItem {
                    Label("词典", systemImage: "character.book.closed")
                }

            modelTab
                .tabItem {
                    Label("模型", systemImage: "cpu")
                }
        }
        .padding(20)
        .frame(width: 560, height: 500)
        .background(SettingsWindowActivator().frame(width: 0, height: 0))
    }
}

private extension SettingsView {
    var generalTab: some View {
        Form {
            Section {
                HStack(alignment: .center, spacing: 16) {
                    Text("识别语言")
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(.secondary)
                    Picker("", selection: $appState.settings.language) {
                        Text("中文").tag("zh")
                        Text("英文").tag("en")
                        Text("自动检测").tag("auto")
                    }
                    .labelsHidden()
                    .frame(maxWidth: 200, alignment: .leading)
                }
            } header: {
                Text("语言设置")
                    .font(.headline)
                    .padding(.bottom, 4)
            }
            .padding(.bottom, 16)
            
            Section {
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
                
                HStack {
                    HotKeyRecorderButton(hotKey: $appState.settings.hotKey)
                }
            } header: {
                Text("快捷键")
                    .font(.headline)
                    .padding(.bottom, 4)
            }
            .padding(.bottom, 16)

            Section {
                if !TextInsertion.isTrustedForInput() {
                    HStack {
                        Button("请求辅助功能权限") {
                            TextInsertion.requestAccessibilityPermission()
                        }
                        Text("用于将转写结果输入到前台应用（模拟粘贴）。")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("辅助功能权限已授权")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("权限")
                    .font(.headline)
                    .padding(.bottom, 4)
            }
        }
        .formStyle(.grouped)
        .scrollDisabled(true)
    }
    
    var hotwordsTab: some View {
        Form {
            Section {
                TextEditor(text: $appState.settings.hotwords)
                    .font(.system(.body, design: .monospaced))
                    .frame(height: 200)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
            } header: {
                Text("词典（同音字优先识别，每行一个词）")
                    .font(.headline)
                    .padding(.bottom, 4)
            } footer: {
                Text("例如：伊智科技")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollDisabled(true)
    }

    var modelTab: some View {
        Form {
            let modelId = appState.settings.modelSize.defaultModelId
            let cachePath = (try? ModelDownloader.getCacheDirectory(for: modelId).path) ?? "-"

            Section {
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
                        .lineLimit(1)
                        .truncationMode(.middle)
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
                        HStack(spacing: 8) {
                            ProgressView(value: progress)
                                .frame(width: 120)
                            
                            if let speed = appState.downloadSpeed {
                                Text(speed)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .frame(width: 80, alignment: .leading)
                                    .lineLimit(1)
                                    .fixedSize(horizontal: true, vertical: false)
                            } else {
                                Spacer().frame(width: 80)
                            }
                        }
                    }
                }
            } header: {
                Text("模型配置")
                    .font(.headline)
                    .padding(.bottom, 4)
            }
        }
        .formStyle(.grouped)
        .scrollDisabled(true)
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
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
            let modifiers = carbonModifiers(from: event.modifierFlags)
            hotKey = HotKey(keyCode: UInt32(event.keyCode), modifiers: modifiers, isEnabled: true)
            stopCapture()
            return nil
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
    func makeNSView(context: Context) -> ActivatorView {
        ActivatorView()
    }

    func updateNSView(_ nsView: ActivatorView, context: Context) {}

    /// 自定义 NSView，在每次被加入窗口时自动激活 App 并置前
    final class ActivatorView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { return }
            // 延迟一个 runloop 确保窗口完全就绪
            DispatchQueue.main.async {
                NSApp.activate(ignoringOtherApps: true)
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
            }
        }
    }
}
