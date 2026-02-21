import Foundation

public enum ModelScopeDownloader {
    private static let maxRetries = 3

    private static func downloadURL(modelId: String, revision: String = "master", file: String) throws -> URL {
        let safeFile = try HuggingFaceDownloader.validatedRemoteFileName(file)

        var components = URLComponents()
        components.scheme = "https"
        components.host = "modelscope.cn"
        components.path = "/api/v1/models/\(modelId)/repo"
        components.queryItems = [
            URLQueryItem(name: "Revision", value: revision),
            URLQueryItem(name: "FilePath", value: safeFile)
        ]

        guard let url = components.url else {
            throw DownloadError.failedToDownload(safeFile)
        }
        return url
    }

    private static func downloadFile(
        url: URL,
        to localPath: URL,
        fileName: String,
        progressHandler: ((Int64, Int64) -> Void)? = nil,
        speedHandler: ((String) -> Void)? = nil
    ) async throws {
        var lastError: Error?
        for attempt in 1...maxRetries {
            do {
                try await downloadFileOnce(url: url, to: localPath, fileName: fileName, progressHandler: progressHandler, speedHandler: speedHandler)
                return
            } catch {
                lastError = error
                if attempt < maxRetries {
                    let delay = UInt64(pow(2.0, Double(attempt - 1))) * 1_000_000_000
                    print("[Download] Retry \(attempt)/\(maxRetries) for \(fileName): \(error.localizedDescription)")
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }
        throw lastError ?? DownloadError.failedToDownload(fileName)
    }

    private static func downloadFileOnce(
        url: URL,
        to localPath: URL,
        fileName: String,
        progressHandler: ((Int64, Int64) -> Void)? = nil,
        speedHandler: ((String) -> Void)? = nil
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let delegate = DownloadDelegate(
                localPath: localPath,
                fileName: fileName,
                continuation: continuation,
                onProgress: progressHandler,
                onSpeed: speedHandler
            )
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 30
            config.timeoutIntervalForResource = 600
            config.waitsForConnectivity = true
            let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
            delegate.session = session
            let task = session.downloadTask(with: url)
            task.resume()
        }
    }

    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil,
        speedHandler: ((String) -> Void)? = nil
    ) async throws {
        var filesToDownload = ["config.json"]
        filesToDownload.append(contentsOf: additionalFiles)

        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            let indexName = "model.safetensors.index.json"
            let indexPath = directory.appendingPathComponent(indexName)

            if !FileManager.default.fileExists(atPath: indexPath.path) {
                let indexURL = try downloadURL(modelId: modelId, file: indexName)
                let session = URLSession(configuration: .default)
                defer { session.finishTasksAndInvalidate() }
                if let (tempURL, indexResponse) = try? await session.download(from: indexURL),
                   let httpResponse = indexResponse as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    try? FileManager.default.moveItem(at: tempURL, to: indexPath)
                }
            }

            var modelFiles: [String] = []
            if FileManager.default.fileExists(atPath: indexPath.path),
               let indexData = try? Data(contentsOf: indexPath),
               let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
               let weightMap = index["weight_map"] as? [String: String],
               !weightMap.isEmpty {
                let uniqueFiles = Set(weightMap.values)
                modelFiles = Array(uniqueFiles).sorted()
            } else {
                try? FileManager.default.removeItem(at: indexPath)
                modelFiles = ["model.safetensors"]
            }

            filesToDownload.append(contentsOf: modelFiles)
        }

        let totalFiles = Double(filesToDownload.count)

        for (index, file) in filesToDownload.enumerated() {
            let safeFile = try HuggingFaceDownloader.validatedRemoteFileName(file)
            let localPath = try HuggingFaceDownloader.validatedLocalPath(directory: directory, fileName: safeFile)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / totalFiles)
                continue
            }

            let fileIndex = Double(index)
            let url = try downloadURL(modelId: modelId, file: safeFile)

            try await downloadFile(url: url, to: localPath, fileName: safeFile, progressHandler: { bytesWritten, totalBytes in
                if totalBytes > 0 {
                    let fileProgress = Double(bytesWritten) / Double(totalBytes)
                    let overall = (fileIndex + fileProgress) / totalFiles
                    progressHandler?(overall)
                }
            }, speedHandler: speedHandler)

            progressHandler?(Double(index + 1) / totalFiles)
        }
    }
}

private class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let localPath: URL
    let fileName: String
    private var continuation: CheckedContinuation<Void, Error>?
    let onProgress: ((Int64, Int64) -> Void)?
    let onSpeed: ((String) -> Void)?
    var session: URLSession?
    
    private var lastBytesWritten: Int64 = 0
    private var lastTime: TimeInterval = 0

    init(
        localPath: URL,
        fileName: String,
        continuation: CheckedContinuation<Void, Error>,
        onProgress: ((Int64, Int64) -> Void)?,
        onSpeed: ((String) -> Void)? = nil
    ) {
        self.localPath = localPath
        self.fileName = fileName
        self.continuation = continuation
        self.onProgress = onProgress
        self.onSpeed = onSpeed
        self.lastTime = Date().timeIntervalSince1970
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress?(totalBytesWritten, totalBytesExpectedToWrite)
        
        let currentTime = Date().timeIntervalSince1970
        let timeElapsed = currentTime - lastTime
        if timeElapsed >= 1.0 { // Update speed roughly every second
            let bytesPerSecond = Double(totalBytesWritten - lastBytesWritten) / timeElapsed
            let mbPerSecond = bytesPerSecond / 1_048_576.0
            
            let speedString: String
            if mbPerSecond > 1.0 {
                speedString = String(format: "%.1f MB/s", mbPerSecond)
            } else {
                let kbPerSecond = bytesPerSecond / 1024.0
                speedString = String(format: "%.0f KB/s", kbPerSecond)
            }
            
            onSpeed?(speedString)
            
            self.lastBytesWritten = totalBytesWritten
            self.lastTime = currentTime
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let fm = FileManager.default
        do {
            if fm.fileExists(atPath: localPath.path) {
                try fm.removeItem(at: localPath)
            }
            try fm.moveItem(at: location, to: localPath)
        } catch {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        defer {
            self.session?.finishTasksAndInvalidate()
            self.session = nil
        }
        if let error {
            continuation?.resume(throwing: error)
        } else if let httpResponse = task.response as? HTTPURLResponse,
                  httpResponse.statusCode != 200 {
            continuation?.resume(throwing: DownloadError.failedToDownload(
                "\(fileName) (HTTP \(httpResponse.statusCode))"))
        } else {
            continuation?.resume()
        }
        continuation = nil
    }
}

