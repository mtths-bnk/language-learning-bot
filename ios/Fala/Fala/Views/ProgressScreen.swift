import SwiftUI

struct ProgressScreen: View {
    @EnvironmentObject private var app: AppModel

    private var report: WeaknessReport { app.weaknessReport }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button {
                    app.goHome()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(FalaTheme.atlantic)
                        .frame(width: 44, height: 44)
                }
                Spacer()
                Text("Progresso")
                    .font(FalaTheme.brand(28, weight: .semibold))
                    .foregroundStyle(FalaTheme.atlantic)
                Spacer()
                Color.clear.frame(width: 44, height: 44)
            }
            .padding(.horizontal, 12)

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 28) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("A sua prática")
                            .font(FalaTheme.body(14, weight: .semibold))
                            .foregroundStyle(FalaTheme.ink.opacity(0.5))
                        Text(summaryLine)
                            .font(FalaTheme.brand(34, weight: .medium))
                            .foregroundStyle(FalaTheme.atlantic)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 18) {
                        MetricBlock(title: "Frases", value: "\(app.progress.totals.utterances)")
                        MetricBlock(title: "Nível", value: app.progress.lastLevel ?? "—")
                        MetricBlock(title: "Erros / frase", value: String(format: "%.1f", app.progress.errorRate))
                        MetricBlock(title: "Vocabulário", value: "\(app.progress.vocabulary.count)")
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Foco recente")
                            .font(FalaTheme.body(14, weight: .semibold))
                            .foregroundStyle(FalaTheme.ink.opacity(0.5))

                        if report.insufficientData {
                            Text(report.message ?? "Continue a praticar.")
                                .font(FalaTheme.body(16))
                                .foregroundStyle(FalaTheme.ink.opacity(0.75))
                        } else {
                            Text("Tendência: \(trendLabel)")
                                .font(FalaTheme.body(18, weight: .medium))
                                .foregroundStyle(FalaTheme.ink)
                            Text("Taxa recente: \(String(format: "%.1f", report.recentErrorRate)) erros/frase")
                                .font(FalaTheme.body(15))
                                .foregroundStyle(FalaTheme.ink.opacity(0.7))

                            if !report.recommendations.isEmpty {
                                Text(report.recommendations.prefix(2).map(prettyRec).joined(separator: " · "))
                                    .font(FalaTheme.body(15, weight: .medium))
                                    .foregroundStyle(FalaTheme.deepTide)
                            }
                        }
                    }

                    if !app.progress.vocabulary.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Palavras")
                                .font(FalaTheme.body(14, weight: .semibold))
                                .foregroundStyle(FalaTheme.ink.opacity(0.5))
                            Text(app.progress.vocabulary.prefix(24).joined(separator: "  ·  "))
                                .font(FalaTheme.body(15))
                                .foregroundStyle(FalaTheme.ink.opacity(0.8))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    Button {
                        app.openPractice()
                    } label: {
                        Text("Continuar a falar")
                            .font(FalaTheme.body(17, weight: .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(FalaTheme.atlantic)
                            .foregroundStyle(FalaTheme.foam)
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 8)
                }
                .padding(.horizontal, 28)
                .padding(.top, 12)
                .padding(.bottom, 40)
            }
        }
    }

    private var summaryLine: String {
        if app.progress.totals.utterances == 0 {
            return "Ainda sem frases. A primeira sessão começa agora."
        }
        return "\(app.progress.totals.utterances) frases guardadas no iPhone."
    }

    private var trendLabel: String {
        switch report.trend {
        case "improving": return "a melhorar"
        case "declining": return "a precisar de revisão"
        default: return "estável"
        }
    }

    private func prettyRec(_ value: String) -> String {
        value.replacingOccurrences(of: "_", with: " ")
    }
}

struct MetricBlock: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(FalaTheme.body(12, weight: .medium))
                .foregroundStyle(FalaTheme.ink.opacity(0.45))
            Text(value)
                .font(FalaTheme.brand(28, weight: .medium))
                .foregroundStyle(FalaTheme.atlantic)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 4)
    }
}
