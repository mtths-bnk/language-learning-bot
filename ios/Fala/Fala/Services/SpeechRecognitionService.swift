import Foundation
import Speech
import AVFoundation

@MainActor
final class SpeechRecognitionService: ObservableObject {
    enum Status: Equatable {
        case idle
        case requestingPermissions
        case ready
        case recording
        case unavailable(String)
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var liveTranscript: String = ""

    private let locale = Locale(identifier: "pt-PT")
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    var isOnDeviceCapable: Bool {
        recognizer?.supportsOnDeviceRecognition == true
    }

    func prepare() async {
        status = .requestingPermissions
        let speechOK = await requestSpeechAuth()
        let micOK = await requestMicAuth()

        guard speechOK, micOK else {
            status = .unavailable("Microfone ou reconhecimento de fala sem permissão.")
            return
        }

        let recognizer = SFSpeechRecognizer(locale: locale)
        self.recognizer = recognizer

        guard let recognizer, recognizer.isAvailable else {
            status = .unavailable("Reconhecimento de português europeu indisponível neste dispositivo.")
            return
        }

        if !recognizer.supportsOnDeviceRecognition {
            status = .unavailable("Este iPhone não tem reconhecimento on-device para pt-PT. Descarregue o idioma em Definições › Geral › Teclado › Ditado.")
            return
        }

        status = .ready
    }

    func start() throws {
        guard status == .ready || status == .idle else { return }
        stopEngineOnly()

        liveTranscript = ""
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        self.request = request

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        task = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    self.liveTranscript = result.bestTranscription.formattedString
                }
                if error != nil {
                    self.stopEngineOnly()
                    if self.status == .recording {
                        self.status = .ready
                    }
                }
            }
        }

        status = .recording
    }

    func stop() -> String {
        let text = liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        request?.endAudio()
        stopEngineOnly()
        task?.cancel()
        task = nil
        request = nil
        status = .ready
        return text
    }

    private func stopEngineOnly() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
    }

    private func requestSpeechAuth() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    private func requestMicAuth() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}
