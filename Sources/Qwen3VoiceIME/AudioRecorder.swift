import AVFoundation
import Foundation

final class AudioRecorder {
    private let engine = AVAudioEngine()
    private let lock = NSLock()

    private var sampleRate: Int = 16000
    private var samples: [Float] = []
    private var isStarted: Bool = false

    func requestMicrophoneAccess() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .denied, .restricted:
            throw RecorderError.microphoneDenied
        case .notDetermined:
            let granted = await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { ok in
                    continuation.resume(returning: ok)
                }
            }
            if !granted {
                throw RecorderError.microphoneDenied
            }
        @unknown default:
            throw RecorderError.microphoneDenied
        }
    }

    func start() throws {
        if isStarted { return }
        samples.removeAll(keepingCapacity: false)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        sampleRate = Int(inputFormat.sampleRate)

        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let data = buffer.floatChannelData else { return }
            let channelCount = Int(buffer.format.channelCount)
            let frameLength = Int(buffer.frameLength)
            if frameLength == 0 { return }

            self.lock.lock()
            defer { self.lock.unlock() }

            if channelCount > 0 {
                let ptr = data[0]
                self.samples.append(contentsOf: UnsafeBufferPointer(start: ptr, count: frameLength))
            }
        }

        engine.prepare()
        try engine.start()
        isStarted = true
    }

    func stop() {
        if !isStarted { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine.reset()
        isStarted = false
    }

    func consumeSamples() -> ([Float], Int) {
        lock.lock()
        defer { lock.unlock() }
        let data = samples
        samples.removeAll(keepingCapacity: false)
        return (data, sampleRate)
    }
}

enum RecorderError: LocalizedError {
    case microphoneDenied

    var errorDescription: String? {
        switch self {
        case .microphoneDenied:
            return "麦克风权限未授权"
        }
    }
}
