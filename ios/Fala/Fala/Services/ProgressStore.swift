import Foundation

final class ProgressStore {
    private let fileName = "learning_progress.json"
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()
    private let decoder = JSONDecoder()

    private var fileURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(fileName)
    }

    func load() -> LearningProgress {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return .empty
        }
        do {
            let data = try Data(contentsOf: fileURL)
            return try decoder.decode(LearningProgress.self, from: data)
        } catch {
            return .empty
        }
    }

    func save(_ progress: LearningProgress) {
        do {
            let data = try encoder.encode(progress)
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            // Local persistence failure should not crash practice.
        }
    }

    func update(progress: LearningProgress, analysis: UtteranceAnalysis, userText: String) -> LearningProgress {
        var next = progress
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        let entry = LearningProgress.SessionEntry(
            timestamp: formatter.string(from: Date()),
            utterance: userText,
            analysis: .init(
                errorCount: analysis.errorCount,
                overallLevel: analysis.overallLevel,
                grammar: analysis.grammar,
                vocabulary: analysis.vocabulary,
                pronunciation: analysis.pronunciation,
                suggestedCorrectedSentence: analysis.suggestedCorrectedSentence
            )
        )
        next.sessions.append(entry)
        next.totals.utterances += 1
        next.totals.errors += analysis.errorCount
        next.lastLevel = analysis.overallLevel

        var vocab = Set(next.vocabulary.map { $0.lowercased() })
        for word in analysis.newWords {
            let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if !trimmed.isEmpty {
                vocab.insert(trimmed)
            }
        }
        next.vocabulary = vocab.sorted()
        return next
    }
}
