import Foundation

public enum ModelDownloader {
    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        try HuggingFaceDownloader.getCacheDirectory(for: modelId, cacheDirName: cacheDirName)
    }

    public static func weightsExist(in directory: URL) -> Bool {
        HuggingFaceDownloader.weightsExist(in: directory)
    }

    public static func downloadWeights(
        source: ModelDownloadSource,
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        switch source {
        case .huggingface:
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: directory,
                additionalFiles: additionalFiles,
                progressHandler: progressHandler
            )
        case .modelscope:
            try await ModelScopeDownloader.downloadWeights(
                modelId: modelId,
                to: directory,
                additionalFiles: additionalFiles,
                progressHandler: progressHandler
            )
        }
    }
}

