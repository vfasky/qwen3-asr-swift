import AppKit
import Combine
import Foundation
import Qwen3ASR
import Qwen3Common

@MainActor
final class AppState: ObservableObject {
    @Published var statusText: String = "就绪"
    @Published var isRecording: Bool = false
    @Published var isBusy: Bool = false
    @Published var downloadProgress: Double? = nil
    @Published var downloadSpeed: String? = nil
    @Published var lastTranscription: String = ""
    @Published var lastError: String? = nil

    @Published var settings = SettingsStore()

    private let recorder = AudioRecorder()
    private let hotKeyManager = GlobalHotKeyManager()
    private var model: Qwen3ASRModel? = nil
    private var cancellables: Set<AnyCancellable> = []
    private var targetApplication: NSRunningApplication? = nil

    init() {
        hotKeyManager.onHotKey = { [weak self] in
            Task { @MainActor in
                await self?.toggleRecordingFromHotKey()
            }
        }

        settings.$hotKey
            .sink { [weak self] hotKey in
                self?.hotKeyManager.register(hotKey: hotKey)
            }
            .store(in: &cancellables)

        Publishers.CombineLatest(settings.$modelSize, settings.$downloadSource)
            .sink { [weak self] _, _ in
                self?.model = nil
            }
            .store(in: &cancellables)

        hotKeyManager.register(hotKey: settings.hotKey)
    }

    var menuBarIconName: String {
        if isRecording { return "mic.fill" }
        if isBusy { return "waveform" }
        return "mic"
    }

    func toggleRecordingFromHotKey() async {
        if isBusy { return }
        if isRecording {
            await stopAndTranscribe()
        } else {
            updateTargetApplication()
            await startRecording()
        }
    }

    func startRecording() async {
        if isBusy || isRecording { return }
        lastError = nil
        updateTargetApplication()
        do {
            try await recorder.requestMicrophoneAccess()
            try recorder.start()
            isRecording = true
            statusText = "录音中… 再按一次热键停止"
        } catch {
            lastError = error.localizedDescription
            statusText = "无法开始录音"
        }
    }

    func stopAndTranscribe() async {
        if isBusy || !isRecording { return }
        isRecording = false
        recorder.stop()
        let (samples, sampleRate) = recorder.consumeSamples()
        if samples.isEmpty {
            statusText = "未捕获到音频"
            return
        }

        isBusy = true
        statusText = "转写中…"
        lastError = nil
        downloadProgress = nil
        downloadSpeed = nil

        do {
            let model = try await ensureModelLoaded()
            let text = await Task.detached(priority: .userInitiated) {
                model.transcribe(audio: samples, sampleRate: sampleRate)
            }.value

            lastTranscription = text
            if let targetApplication, targetApplication.bundleIdentifier != Bundle.main.bundleIdentifier {
                targetApplication.activate(options: [.activateAllWindows])
                try? await Task.sleep(nanoseconds: 80_000_000)
            }
            let insertionResult = await attemptInsertion(text: text)
            switch insertionResult {
            case .success:
                statusText = "已转写并输入"
                lastError = nil
            case .failed(let message):
                statusText = "已转写但未自动输入"
                lastError = message
            }
        } catch {
            lastError = error.localizedDescription
            statusText = "转写失败"
        }

        isBusy = false
        downloadProgress = nil
        downloadSpeed = nil
    }

    func quit() {
        NSApp.terminate(nil)
    }

    func prepareModel() async {
        if isBusy { return }
        isBusy = true
        statusText = "准备模型…"
        lastError = nil
        downloadProgress = nil
        downloadSpeed = nil
        do {
            _ = try await ensureModelLoaded()
            statusText = "模型已就绪"
        } catch {
            lastError = error.localizedDescription
            statusText = "模型准备失败"
        }
        isBusy = false
        downloadProgress = nil
        downloadSpeed = nil
    }

    private func ensureModelLoaded() async throws -> Qwen3ASRModel {
        if let model { return model }

        let modelId = settings.modelSize.defaultModelId
        let source = settings.downloadSource

        let loaded = try await Qwen3ASRModel.fromPretrained(
            modelId: modelId,
            source: source,
            progressHandler: { [weak self] progress, message in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.statusText = message
                }
            },
            speedHandler: { [weak self] speed in
                Task { @MainActor in
                    self?.downloadSpeed = speed
                }
            }
        )
        self.model = loaded
        return loaded
    }

    private func updateTargetApplication() {
        let frontmost = NSWorkspace.shared.frontmostApplication
        if frontmost?.bundleIdentifier != Bundle.main.bundleIdentifier {
            targetApplication = frontmost
        }
    }

    private func attemptInsertion(text: String) async -> TextInsertion.Result {
        let activeApp = targetApplication ?? NSWorkspace.shared.frontmostApplication
        let isVSCode = activeApp?.bundleIdentifier == "com.microsoft.VSCode"
        let isElectron = isVSCode || isElectronApp(activeApp)

        let pid = targetApplication?.processIdentifier ?? activeApp?.processIdentifier

        // Electron 应用：直接走剪贴板粘贴路径，不做无意义的 AX 重试
        if isElectron, let pid {
            return await TextInsertion.pasteInsert(
                text: text,
                targetPID: pid,
                preferMenuPaste: true
            )
        }

        // 原生应用：尝试辅助功能直接写入，带重试
        let delays: [UInt64] = [0, 120_000_000, 200_000_000, 320_000_000]
        var lastResult: TextInsertion.Result = .failed("未知错误")
        for delay in delays {
            if delay > 0 {
                try? await Task.sleep(nanoseconds: delay)
            }
            if let targetApplication, targetApplication.bundleIdentifier != Bundle.main.bundleIdentifier {
                targetApplication.activate(options: [.activateAllWindows])
            }
            let currentPID = pid ?? NSWorkspace.shared.frontmostApplication?.processIdentifier
            lastResult = TextInsertion.insert(text: text, targetPID: currentPID)
            if case .success = lastResult {
                return lastResult
            }
        }

        // 原生应用 AX 写入也全部失败，最终降级到剪贴板粘贴
        if let pid {
            return await TextInsertion.pasteInsert(text: text, targetPID: pid)
        }
        return lastResult
    }

    private func isElectronApp(_ app: NSRunningApplication?) -> Bool {
        guard let app else { return false }
        if app.bundleIdentifier?.lowercased().contains("electron") == true {
            return true
        }
        if let bundleURL = app.bundleURL {
            let frameworkPath = bundleURL.appendingPathComponent("Contents/Frameworks/Electron Framework.framework")
            if FileManager.default.fileExists(atPath: frameworkPath.path) {
                return true
            }
            let helperPath = bundleURL.appendingPathComponent("Contents/Frameworks/Electron Helper.app")
            if FileManager.default.fileExists(atPath: helperPath.path) {
                return true
            }
        }
        return false
    }
}
