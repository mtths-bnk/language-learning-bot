import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

struct UtteranceAnalysis: Codable, Equatable, Sendable {
    var grammar: String?
    var vocabulary: String?
    var pronunciation: String?
    var overallLevel: String
    var errorCount: Int
    var newWords: [String]
    var suggestedCorrectedSentence: String

    static let blank = UtteranceAnalysis(
        grammar: nil,
        vocabulary: nil,
        pronunciation: nil,
        overallLevel: "A2",
        errorCount: 0,
        newWords: [],
        suggestedCorrectedSentence: ""
    )
}

#if canImport(FoundationModels)
@available(iOS 26.0, *)
@Generable
struct GenerableUtteranceAnalysis {
    @Guide(description: "Brief grammar feedback in Portuguese, or null if correct")
    var grammar: String?

    @Guide(description: "Brief vocabulary feedback in Portuguese, or null if correct")
    var vocabulary: String?

    @Guide(description: "Brief pronunciation tip in Portuguese, or null if unknown")
    var pronunciation: String?

    @Guide(description: "CEFR level estimate: A1, A2, B1, B2, C1, or C2")
    var overallLevel: String

    @Guide(description: "Number of distinct learner errors detected")
    var errorCount: Int

    @Guide(description: "New useful Portuguese words from the utterance")
    var newWords: [String]

    @Guide(description: "Corrected version of the learner utterance in Portuguese")
    var suggestedCorrectedSentence: String

    func asAnalysis() -> UtteranceAnalysis {
        UtteranceAnalysis(
            grammar: grammar,
            vocabulary: vocabulary,
            pronunciation: pronunciation,
            overallLevel: overallLevel.isEmpty ? "A2" : overallLevel,
            errorCount: max(0, errorCount),
            newWords: newWords.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty },
            suggestedCorrectedSentence: suggestedCorrectedSentence
        )
    }
}
#endif
