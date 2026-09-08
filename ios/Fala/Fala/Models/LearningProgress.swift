import Foundation

struct LearningProgress: Codable, Equatable {
    struct Totals: Codable, Equatable {
        var utterances: Int
        var errors: Int
    }

    struct SessionEntry: Codable, Equatable, Identifiable {
        var id: String { timestamp }
        var timestamp: String
        var utterance: String
        var analysis: SessionAnalysis
    }

    struct SessionAnalysis: Codable, Equatable {
        var errorCount: Int
        var overallLevel: String?
        var grammar: String?
        var vocabulary: String?
        var pronunciation: String?
        var suggestedCorrectedSentence: String
    }

    var sessions: [SessionEntry]
    var totals: Totals
    var lastLevel: String?
    var vocabulary: [String]

    static let empty = LearningProgress(
        sessions: [],
        totals: Totals(utterances: 0, errors: 0),
        lastLevel: nil,
        vocabulary: []
    )

    var errorRate: Double {
        guard totals.utterances > 0 else { return 0 }
        return Double(totals.errors) / Double(totals.utterances)
    }
}

struct WeaknessReport: Equatable {
    var insufficientData: Bool
    var message: String?
    var recentErrorRate: Double
    var trend: String
    var mainWeaknesses: [String]
    var grammarIssues: [(type: String, frequency: Int)]
    var recommendations: [String]

    static let empty = WeaknessReport(
        insufficientData: true,
        message: "Preciso de mais dados para analisar suas fraquezas. Continue praticando!",
        recentErrorRate: 0,
        trend: "stable",
        mainWeaknesses: [],
        grammarIssues: [],
        recommendations: []
    )
}
