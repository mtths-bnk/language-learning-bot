import Foundation
import Combine

@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable {
        case home
        case practice
        case progress
    }

    enum PracticeState: Equatable {
        case idle
        case listening
        case thinking
        case speaking
        case error(String)
    }

    @Published var phase: Phase = .home
    @Published var practiceState: PracticeState = .idle
    @Published var turns: [ConversationTurn] = []
    @Published var progress: LearningProgress = .empty
    @Published var liveTranscript: String = ""
    @Published var statusBanner: String?

    let speech = SpeechRecognitionService()
    let tts = TextToSpeechService()
    let languageModel = OnDeviceLanguageModel()

    private let store = ProgressStore()
    private var cancellables = Set<AnyCancellable>()

    init() {
        progress = store.load()
        languageModel.refreshAvailability()

        speech.$liveTranscript
            .receive(on: RunLoop.main)
            .sink { [weak self] value in
                self?.liveTranscript = value
            }
            .store(in: &cancellables)
    }

    var weaknessReport: WeaknessReport {
        WeaknessAnalyzer.analyze(progress)
    }

    var modelStatusText: String {
        switch languageModel.availability {
        case .checking:
            return "A verificar modelo on-device…"
        case .available:
            return "Apple Intelligence · on-device"
        case .unavailable(let reason):
            return reason
        }
    }

    func openPractice() {
        phase = .practice
        Task {
            await speech.prepare()
            languageModel.refreshAvailability()
        }
    }

    func openProgress() {
        phase = .progress
    }

    func goHome() {
        phase = .home
        practiceState = .idle
        speech.stop()
        tts.stop()
    }

    func startListening() {
        Task {
            if case .unavailable(let message) = speech.status {
                practiceState = .error(message)
                return
            }
            if speech.status != .ready {
                await speech.prepare()
            }
            do {
                tts.stop()
                try speech.start()
                practiceState = .listening
                statusBanner = nil
            } catch {
                practiceState = .error("Não foi possível gravar: \(error.localizedDescription)")
            }
        }
    }

    func stopListeningAndProcess() {
        guard practiceState == .listening else { return }
        let text = speech.stop()
        liveTranscript = text

        guard !text.isEmpty else {
            practiceState = .error("Não ouvi nada. Toque e fale em português.")
            return
        }

        Task {
            await processUtterance(text)
        }
    }

    func processTypedUtterance(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        await processUtterance(trimmed)
    }

    private func processUtterance(_ text: String) async {
        practiceState = .thinking
        let userTurn = ConversationTurn(role: .user, text: text)
        turns.append(userTurn)

        let analysis = await languageModel.analyze(utterance: text)
        progress = store.update(progress: progress, analysis: analysis, userText: text)
        store.save(progress)

        let context = WeaknessAnalyzer.weaknessContext(from: WeaknessAnalyzer.analyze(progress))
        let reply = await languageModel.respond(to: text, history: turns, weaknessContext: context)

        let assistantTurn = ConversationTurn(role: .assistant, text: reply, analysis: analysis)
        turns.append(assistantTurn)

        practiceState = .speaking
        tts.speak(reply)

        // Return to idle shortly after speech starts; speaking flag tracked by TTS.
        try? await Task.sleep(nanoseconds: 400_000_000)
        if practiceState == .speaking {
            practiceState = .idle
        }
    }

    func resetConversation() {
        turns.removeAll()
        languageModel.resetDialogMemory()
        practiceState = .idle
        statusBanner = "Nova conversa."
    }
}
