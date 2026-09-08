import SwiftUI

struct PracticeView: View {
    @EnvironmentObject private var app: AppModel
    @State private var typedFallback = ""
    @State private var showTypeInput = false

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollViewReader { proxy in
                ScrollView(showsIndicators: false) {
                    LazyVStack(alignment: .leading, spacing: 18) {
                        Text(app.modelStatusText)
                            .font(FalaTheme.body(12, weight: .medium))
                            .foregroundStyle(FalaTheme.ink.opacity(0.5))
                            .padding(.top, 8)

                        if app.turns.isEmpty {
                            emptyState
                        }

                        ForEach(app.turns) { turn in
                            TurnBubble(turn: turn)
                                .id(turn.id)
                        }

                        if app.practiceState == .listening, !app.liveTranscript.isEmpty {
                            Text(app.liveTranscript)
                                .font(FalaTheme.body(17))
                                .foregroundStyle(FalaTheme.atlantic.opacity(0.7))
                                .id("live")
                        }

                        if case .error(let message) = app.practiceState {
                            Text(message)
                                .font(FalaTheme.body(14, weight: .medium))
                                .foregroundStyle(FalaTheme.coralSignal)
                                .id("error")
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.bottom, 24)
                }
                .onChange(of: app.turns.count) { _, _ in
                    if let last = app.turns.last?.id {
                        withAnimation {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }

            controlBar
        }
    }

    private var header: some View {
        HStack {
            Button {
                app.goHome()
            } label: {
                Image(systemName: "chevron.backward")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(FalaTheme.atlantic)
                    .frame(width: 44, height: 44)
            }

            Spacer()

            Text("Fala")
                .font(FalaTheme.brand(28, weight: .semibold))
                .foregroundStyle(FalaTheme.atlantic)

            Spacer()

            Button {
                app.resetConversation()
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(FalaTheme.deepTide)
                    .frame(width: 44, height: 44)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Diga uma frase em português.")
                .font(FalaTheme.body(20, weight: .semibold))
                .foregroundStyle(FalaTheme.ink)
            Text("Mantenha premido o microfone, fale, e solte para obter correção e uma pergunta de seguimento.")
                .font(FalaTheme.body(15))
                .foregroundStyle(FalaTheme.ink.opacity(0.6))
        }
        .padding(.top, 24)
    }

    private var controlBar: some View {
        VStack(spacing: 16) {
            if showTypeInput {
                HStack(spacing: 10) {
                    TextField("Escrever em português…", text: $typedFallback, axis: .vertical)
                        .font(FalaTheme.body(16))
                        .lineLimit(1...3)
                        .padding(12)
                        .background(FalaTheme.foam.opacity(0.9))
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                    Button("Enviar") {
                        let text = typedFallback
                        typedFallback = ""
                        Task { await app.processTypedUtterance(text) }
                    }
                    .font(FalaTheme.body(15, weight: .semibold))
                    .foregroundStyle(FalaTheme.atlantic)
                    .disabled(app.practiceState == .thinking)
                }
                .padding(.horizontal, 24)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            HStack(alignment: .center, spacing: 28) {
                Button {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                        showTypeInput.toggle()
                    }
                } label: {
                    Image(systemName: "keyboard")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundStyle(FalaTheme.deepTide)
                        .frame(width: 48, height: 48)
                }

                HoldToTalkButton(
                    isListening: app.practiceState == .listening,
                    isBusy: app.practiceState == .thinking || app.tts.isSpeaking
                ) {
                    app.startListening()
                } onReleased: {
                    app.stopListeningAndProcess()
                }

                Button {
                    app.openProgress()
                } label: {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundStyle(FalaTheme.deepTide)
                        .frame(width: 48, height: 48)
                }
            }
            .padding(.bottom, 28)
        }
        .padding(.top, 8)
        .background(
            LinearGradient(
                colors: [FalaTheme.foam.opacity(0), FalaTheme.foam.opacity(0.95)],
                startPoint: .top,
                endPoint: .bottom
            )
            .allowsHitTesting(false)
        )
    }
}

struct TurnBubble: View {
    let turn: ConversationTurn

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(turn.role == .user ? "Você" : "Fala")
                .font(FalaTheme.body(12, weight: .semibold))
                .foregroundStyle(turn.role == .user ? FalaTheme.deepTide : FalaTheme.atlantic)

            Text(displayText)
                .font(FalaTheme.body(turn.role == .assistant ? 18 : 17, weight: turn.role == .assistant ? .medium : .regular))
                .foregroundStyle(FalaTheme.ink)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let analysis = turn.analysis, turn.role == .assistant {
                AnalysisStrip(analysis: analysis)
            }
        }
        .padding(.vertical, 4)
    }

    private var displayText: String {
        turn.text
            .replacingOccurrences(of: "---", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct AnalysisStrip: View {
    let analysis: UtteranceAnalysis

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Nível \(analysis.overallLevel) · \(analysis.errorCount) erro\(analysis.errorCount == 1 ? "" : "s")")
                .font(FalaTheme.body(12, weight: .semibold))
                .foregroundStyle(FalaTheme.atlantic.opacity(0.7))

            if !analysis.suggestedCorrectedSentence.isEmpty, analysis.errorCount > 0 {
                Text(analysis.suggestedCorrectedSentence)
                    .font(FalaTheme.body(13, weight: .medium))
                    .foregroundStyle(FalaTheme.deepTide)
            }
        }
        .padding(.top, 2)
    }
}

struct HoldToTalkButton: View {
    let isListening: Bool
    let isBusy: Bool
    let onPressed: () -> Void
    let onReleased: () -> Void

    @State private var pulse = false

    var body: some View {
        ZStack {
            Circle()
                .fill(FalaTheme.seaGlass.opacity(isListening ? 0.45 : 0.2))
                .frame(width: 118, height: 118)
                .scaleEffect(pulse && isListening ? 1.12 : 1)
                .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: pulse)

            Circle()
                .fill(isListening ? FalaTheme.coralSignal : FalaTheme.atlantic)
                .frame(width: 84, height: 84)
                .shadow(color: FalaTheme.atlantic.opacity(0.22), radius: 16, y: 8)

            Group {
                if isBusy && !isListening {
                    ProgressView()
                        .tint(FalaTheme.foam)
                } else {
                    Image(systemName: isListening ? "waveform" : "mic.fill")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(FalaTheme.foam)
                        .symbolEffect(.variableColor, isActive: isListening)
                }
            }
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !isBusy || isListening else { return }
                    if !isListening { onPressed() }
                }
                .onEnded { _ in
                    if isListening { onReleased() }
                }
        )
        .disabled(isBusy && !isListening)
        .accessibilityLabel(isListening ? "A ouvir" : "Manter premido para falar")
        .onAppear { pulse = true }
        .onChange(of: isListening) { _, listening in
            pulse = listening
        }
    }
}
