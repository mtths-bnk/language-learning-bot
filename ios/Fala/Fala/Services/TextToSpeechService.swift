import Foundation
import AVFoundation

@MainActor
final class TextToSpeechService: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false

    private let synthesizer = AVSpeechSynthesizer()
    private let voiceLanguage = "pt-PT"

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String) {
        let cleaned = text
            .replacingOccurrences(of: "---", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }

        synthesizer.stopSpeaking(at: .immediate)

        let utterance = AVSpeechUtterance(string: cleaned)
        utterance.voice = preferredPortugueseVoice()
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.92
        utterance.pitchMultiplier = 1.02
        utterance.preUtteranceDelay = 0.05

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            // Continue; synthesis may still work with current session.
        }

        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }

    private func preferredPortugueseVoice() -> AVSpeechSynthesisVoice? {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("pt") }
        if let european = voices.first(where: { $0.language == voiceLanguage && $0.quality == .enhanced }) {
            return european
        }
        if let european = voices.first(where: { $0.language == voiceLanguage }) {
            return european
        }
        return AVSpeechSynthesisVoice(language: voiceLanguage) ?? voices.first
    }
}
