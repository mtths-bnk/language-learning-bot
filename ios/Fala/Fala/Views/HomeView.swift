import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var app: AppModel
    @State private var appeared = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)

            VStack(alignment: .leading, spacing: 18) {
                Text("Fala")
                    .font(FalaTheme.brand(64, weight: .semibold))
                    .foregroundStyle(FalaTheme.atlantic)
                    .opacity(appeared ? 1 : 0)
                    .offset(y: appeared ? 0 : 12)

                Text("Português europeu, só no seu iPhone.")
                    .font(FalaTheme.body(22, weight: .medium))
                    .foregroundStyle(FalaTheme.ink.opacity(0.88))
                    .fixedSize(horizontal: false, vertical: true)
                    .opacity(appeared ? 1 : 0)
                    .offset(y: appeared ? 0 : 10)

                Text("Fale. Ouça correções. Tudo on-device.")
                    .font(FalaTheme.body(16))
                    .foregroundStyle(FalaTheme.ink.opacity(0.62))
                    .opacity(appeared ? 1 : 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 28)

            Spacer()

            ZStack {
                Circle()
                    .fill(FalaTheme.seaGlass.opacity(0.35))
                    .frame(width: 280, height: 280)
                    .blur(radius: 30)
                    .waveMotion()

                Image(systemName: "waveform")
                    .font(.system(size: 84, weight: .light))
                    .foregroundStyle(FalaTheme.atlantic.opacity(0.85))
                    .symbolEffect(.variableColor.iterative, options: .repeating, isActive: appeared)
            }
            .frame(maxWidth: .infinity)
            .padding(.bottom, 28)

            Spacer()

            VStack(spacing: 14) {
                Button {
                    app.openPractice()
                } label: {
                    Text("Começar a falar")
                        .font(FalaTheme.body(18, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 18)
                        .background(FalaTheme.atlantic)
                        .foregroundStyle(FalaTheme.foam)
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                }
                .buttonStyle(.plain)

                Button {
                    app.openProgress()
                } label: {
                    Text("Ver progresso")
                        .font(FalaTheme.body(16, weight: .medium))
                        .foregroundStyle(FalaTheme.deepTide)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 28)
            .padding(.bottom, 36)
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 16)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.7)) {
                appeared = true
            }
        }
    }
}
