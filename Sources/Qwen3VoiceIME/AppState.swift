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
    @Published var lastTranscription: String = ""
    @Published var lastError: String? = nil

    @Published var settings = SettingsStore()

    private let recorder = AudioRecorder()
    private let hotKeyManager = GlobalHotKeyManager()
    private var model: Qwen3ASRModel? = nil
    private var cancellables: Set<AnyCancellable> = []

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
            await startRecording()
        }
    }

    func startRecording() async {
        if isBusy || isRecording { return }
        lastError = nil
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

        do {
            let model = try await ensureModelLoaded()
            let text = await Task.detached(priority: .userInitiated) {
                model.transcribe(audio: samples, sampleRate: sampleRate)
            }.value

            lastTranscription = text
            statusText = "已转写并输入"
            TextInsertion.insert(text: text)
        } catch {
            lastError = error.localizedDescription
            statusText = "转写失败"
        }

        isBusy = false
        downloadProgress = nil
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
        do {
            _ = try await ensureModelLoaded()
            statusText = "模型已就绪"
        } catch {
            lastError = error.localizedDescription
            statusText = "模型准备失败"
        }
        isBusy = false
        downloadProgress = nil
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
            }
        )
        self.model = loaded
        return loaded
    }
}
