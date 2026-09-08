import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

enum ModelAvailability: Equatable {
    case available
    case unavailable(String)
    case checking
}

@MainActor
final class OnDeviceLanguageModel: ObservableObject {
    @Published private(set) var availability: ModelAvailability = .checking

    private var dialogInstructions: String {
        """
        You are a friendly AI assistant that helps the user learn European Portuguese. \
        You are precise in grammar and vocabulary. If the user makes a mistake, correct them \
        and provide a short explanation. Keep it short and crisp. Always respond in Portuguese!

        Use exactly this format:
        ---
        <A response correcting mistakes with a quick explanation in Portuguese (max. 2 sentences).>
        <Your quick response containing a follow-up question in Portuguese (max. 2 sentences).>
        ---
        """
    }

#if canImport(FoundationModels)
    private var dialogSession: LanguageModelSession?
#endif

    func refreshAvailability() {
#if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            let model = SystemLanguageModel.default
            switch model.availability {
            case .available:
                availability = .available
                if dialogSession == nil {
                    dialogSession = LanguageModelSession(instructions: dialogInstructions)
                }
            case .unavailable(let reason):
                availability = .unavailable(unavailableMessage(reason))
            @unknown default:
                availability = .unavailable("Modelo on-device indisponível.")
            }
            return
        }
#endif
        availability = .unavailable("Requer iOS 26+ com Apple Intelligence para o modelo on-device.")
    }

    func analyze(utterance: String) async -> UtteranceAnalysis {
#if canImport(FoundationModels)
        if #available(iOS 26.0, *), case .available = availability {
            do {
                let session = LanguageModelSession(
                    instructions: """
                    You are a precise European Portuguese tutor. Analyze the learner utterance \
                    and return structured feedback only.
                    """
                )
                let response = try await session.respond(
                    to: "Analyze this Portuguese learner utterance: \(utterance)",
                    generating: GenerableUtteranceAnalysis.self
                )
                return response.content.asAnalysis()
            } catch {
                return HeuristicTutor.analyze(utterance)
            }
        }
#endif
        return HeuristicTutor.analyze(utterance)
    }

    func respond(to userText: String, history: [ConversationTurn], weaknessContext: String) async -> String {
#if canImport(FoundationModels)
        if #available(iOS 26.0, *), case .available = availability {
            do {
                if dialogSession == nil {
                    dialogSession = LanguageModelSession(instructions: dialogInstructions)
                }
                let historyText = history.suffix(8).map { turn in
                    let who = turn.role == .user ? "User" : "Assistant"
                    return "\(who): \(turn.text)"
                }.joined(separator: "\n")

                let prompt = """
                \(weaknessContext)

                The conversation transcript is as follows:
                \(historyText)

                And here is the user's statement in Portuguese: \(userText)
                Your response:
                """

                let response = try await dialogSession!.respond(to: prompt)
                return sanitize(response.content)
            } catch {
                return HeuristicTutor.respond(to: userText, analysis: HeuristicTutor.analyze(userText))
            }
        }
#endif
        return HeuristicTutor.respond(to: userText, analysis: HeuristicTutor.analyze(userText))
    }

    func resetDialogMemory() {
#if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            dialogSession = LanguageModelSession(instructions: dialogInstructions)
        }
#endif
    }

    private func sanitize(_ text: String) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("Assistant:") {
            value = String(value.dropFirst("Assistant:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return value
    }

#if canImport(FoundationModels)
    @available(iOS 26.0, *)
    private func unavailableMessage(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
        switch reason {
        case .deviceNotEligible:
            return "Este iPhone não é elegível para Apple Intelligence."
        case .appleIntelligenceNotEnabled:
            return "Ative a Apple Intelligence em Definições para usar o tutor on-device."
        case .modelNotReady:
            return "O modelo ainda está a descarregar. Tente novamente em breve."
        @unknown default:
            return "Modelo on-device indisponível."
        }
    }
#endif
}

enum HeuristicTutor {
    static func analyze(_ text: String) -> UtteranceAnalysis {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var errors = 0
        var grammar: String?
        var vocabulary: String?
        var suggestion = trimmed

        let lower = trimmed.lowercased()
        if lower.contains(" eu tem ") || lower.hasPrefix("eu tem ") {
            errors += 1
            grammar = "Use «eu tenho» (verbo ter, 1.ª pessoa)."
            suggestion = trimmed.replacingOccurrences(of: "tem", with: "tenho", options: [.caseInsensitive])
        }
        if lower.contains(" eu vai ") || lower.hasPrefix("eu vai ") {
            errors += 1
            grammar = [grammar, "Use «eu vou» (verbo ir)."].compactMap { $0 }.joined(separator: " ")
            suggestion = suggestion.replacingOccurrences(of: "vai", with: "vou", options: [.caseInsensitive])
        }
        if lower.contains(" mais melhor") {
            errors += 1
            grammar = [grammar, "Evite dupla comparação: diga apenas «melhor»."].compactMap { $0 }.joined(separator: " ")
            suggestion = suggestion.replacingOccurrences(of: "mais melhor", with: "melhor", options: [.caseInsensitive])
        }
        if trimmed.count < 4 {
            vocabulary = "Tente uma frase completa."
            errors += 1
        }

        let words = trimmed
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .map { $0.lowercased() }
            .filter { $0.count > 3 }

        return UtteranceAnalysis(
            grammar: grammar,
            vocabulary: vocabulary,
            pronunciation: nil,
            overallLevel: errors == 0 ? "B1" : "A2",
            errorCount: errors,
            newWords: Array(words.prefix(4)),
            suggestedCorrectedSentence: suggestion
        )
    }

    static func respond(to text: String, analysis: UtteranceAnalysis) -> String {
        let correction: String
        if analysis.errorCount == 0 {
            correction = "Muito bem — a frase está clara e natural."
        } else if let grammar = analysis.grammar {
            correction = grammar
        } else if let vocabulary = analysis.vocabulary {
            correction = vocabulary
        } else {
            correction = "Quase! Experimente: \(analysis.suggestedCorrectedSentence)"
        }

        let followUps = [
            "O que gosta de fazer ao fim de semana?",
            "Pode descrever o seu dia em duas frases?",
            "Qual é a sua comida portuguesa favorita?",
            "Onde gostaria de viajar em Portugal?"
        ]
        let followUp = followUps[abs(text.hashValue) % followUps.count]
        return "---\n\(correction)\n\(followUp)\n---"
    }
}
