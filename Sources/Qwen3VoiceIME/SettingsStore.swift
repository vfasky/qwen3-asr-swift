import Foundation
import Combine
import Carbon
import Qwen3Common

enum ASRModelChoice: String, CaseIterable, Codable {
    case small06B
    case large17B

    var displayName: String {
        switch self {
        case .small06B: return "0.6B"
        case .large17B: return "1.7B"
        }
    }

    var defaultModelId: String {
        switch self {
        case .small06B: return "mlx-community/Qwen3-ASR-0.6B-4bit"
        case .large17B: return "mlx-community/Qwen3-ASR-1.7B-8bit"
        }
    }
}

struct HotKey: Codable, Equatable {
    var keyCode: UInt32
    var modifiers: UInt32
    var isEnabled: Bool

    static let `default` = HotKey(keyCode: 12, modifiers: UInt32(controlKey | shiftKey), isEnabled: true)
}

@MainActor
final class SettingsStore: ObservableObject {
    @Published var hotKey: HotKey
    @Published var modelSize: ASRModelChoice
    @Published var downloadSource: ModelDownloadSource

    private let defaults = UserDefaults.standard
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init() {
        if let data = defaults.data(forKey: "hotKey"),
           let decoded = try? decoder.decode(HotKey.self, from: data) {
            self.hotKey = decoded
        } else {
            self.hotKey = .default
        }

        if let raw = defaults.string(forKey: "modelSize"),
           let value = ASRModelChoice(rawValue: raw) {
            self.modelSize = value
        } else {
            self.modelSize = .large17B
        }

        if let raw = defaults.string(forKey: "downloadSource"),
           let value = ModelDownloadSource(rawValue: raw) {
            self.downloadSource = value
        } else {
            self.downloadSource = .modelscope
        }

        $hotKey
            .sink { [weak self] value in
                guard let self else { return }
                if let data = try? self.encoder.encode(value) {
                    self.defaults.set(data, forKey: "hotKey")
                }
            }
            .store(in: &cancellables)

        $modelSize
            .sink { [weak self] value in
                self?.defaults.set(value.rawValue, forKey: "modelSize")
            }
            .store(in: &cancellables)

        $downloadSource
            .sink { [weak self] value in
                self?.defaults.set(value.rawValue, forKey: "downloadSource")
            }
            .store(in: &cancellables)
    }

    private var cancellables: Set<AnyCancellable> = []
}
