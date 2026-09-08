import Foundation

enum WeaknessAnalyzer {
    static func analyze(_ progress: LearningProgress) -> WeaknessReport {
        let sessions = progress.sessions
        guard sessions.count >= 3 else {
            return .empty
        }

        let recentCutoff = Date().addingTimeInterval(-7 * 24 * 60 * 60)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        var recent: [(date: Date, entry: LearningProgress.SessionEntry)] = []
        var all: [(date: Date, entry: LearningProgress.SessionEntry)] = []

        for entry in sessions.suffix(10) {
            let date = formatter.date(from: entry.timestamp) ?? .distantPast
            let pair = (date, entry)
            all.append(pair)
            if date >= recentCutoff {
                recent.append(pair)
            }
        }

        if recent.isEmpty {
            recent = Array(all.suffix(10))
        }

        var report = WeaknessReport(
            insufficientData: false,
            message: nil,
            recentErrorRate: 0,
            trend: "stable",
            mainWeaknesses: [],
            grammarIssues: [],
            recommendations: []
        )

        if !recent.isEmpty {
            let totalErrors = recent.reduce(0) { $0 + $1.entry.analysis.errorCount }
            report.recentErrorRate = Double(totalErrors) / Double(recent.count)

            if recent.count >= 4 {
                let mid = recent.count / 2
                let first = Double(recent.prefix(mid).reduce(0) { $0 + $1.entry.analysis.errorCount }) / Double(mid)
                let secondCount = recent.count - mid
                let second = Double(recent.suffix(secondCount).reduce(0) { $0 + $1.entry.analysis.errorCount }) / Double(secondCount)
                if second < first * 0.8 {
                    report.trend = "improving"
                } else if second > first * 1.2 {
                    report.trend = "declining"
                }
            }
        }

        var grammarCounter: [String: Int] = [:]
        for item in recent {
            guard let grammar = item.entry.analysis.grammar?.lowercased(),
                  !["correct", "none", "null", ""].contains(grammar) else { continue }

            if grammar.contains("verbo") || grammar.contains("conjug") || grammar.contains("tempo") {
                grammarCounter["verb_conjugation", default: 0] += 1
            }
            if grammar.contains("gênero") || grammar.contains("genero") || grammar.contains("masculino") || grammar.contains("feminino") {
                grammarCounter["gender_agreement", default: 0] += 1
            }
            if grammar.contains("prepos") {
                grammarCounter["prepositions", default: 0] += 1
            }
            if grammar.contains("artigo") {
                grammarCounter["articles", default: 0] += 1
            }
            if grammar.contains("plural") || grammar.contains("singular") || grammar.contains("concord") {
                grammarCounter["number_agreement", default: 0] += 1
            }
        }

        report.grammarIssues = grammarCounter
            .map { (type: $0.key, frequency: $0.value) }
            .sorted { $0.frequency > $1.frequency }
            .prefix(3)
            .map { $0 }

        if report.recentErrorRate > 2.0 {
            report.recommendations.append("focus_on_accuracy")
        }
        if report.trend == "declining" {
            report.recommendations.append("review_fundamentals")
        }
        if report.grammarIssues.contains(where: { $0.type == "verb_conjugation" }) {
            report.recommendations.append("practice_verb_conjugation")
        }
        if report.grammarIssues.contains(where: { $0.type == "gender_agreement" }) {
            report.recommendations.append("study_noun_genders")
        }
        if report.recentErrorRate > 1.5 {
            report.mainWeaknesses.append("high_error_rate")
        }
        report.mainWeaknesses.append(contentsOf: report.grammarIssues.prefix(2).map(\.type))

        return report
    }

    static func weaknessContext(from report: WeaknessReport) -> String {
        guard !report.insufficientData else { return "" }

        var parts: [String] = []
        if report.recentErrorRate > 2.0 {
            parts.append("O usuário tem uma taxa de erro alta (\(String(format: "%.1f", report.recentErrorRate)) erros por frase em média)")
        } else if report.recentErrorRate > 1.0 {
            parts.append("O usuário comete alguns erros (\(String(format: "%.1f", report.recentErrorRate)) erros por frase em média)")
        }

        switch report.trend {
        case "declining":
            parts.append("O progresso tem diminuído recentemente")
        case "improving":
            parts.append("O usuário está melhorando")
        default:
            break
        }

        let issueMap = [
            "verb_conjugation": "conjugação de verbos",
            "gender_agreement": "concordância de gênero",
            "prepositions": "uso de preposições",
            "articles": "uso de artigos",
            "number_agreement": "concordância de número"
        ]
        if let main = report.grammarIssues.first, let label = issueMap[main.type] {
            parts.append("Dificuldade principal: \(label)")
        }

        let recMap = [
            "focus_on_accuracy": "focar na precisão",
            "review_fundamentals": "revisar os fundamentos",
            "practice_verb_conjugation": "praticar conjugação de verbos",
            "study_noun_genders": "estudar géneros dos substantivos"
        ]
        let recs = report.recommendations.prefix(2).compactMap { recMap[$0] }
        if !recs.isEmpty {
            parts.append("Recomendações: \(recs.joined(separator: ", "))")
        }

        guard !parts.isEmpty else { return "" }
        return "\n\nCONTEXTO DO PROGRESSO: \(parts.joined(separator: " | ")). Use essas informações para direcionar a conversa e ajudar com as dificuldades específicas."
    }
}
