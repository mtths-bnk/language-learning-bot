import SwiftUI

enum FalaTheme {
    static let atlantic = Color(red: 0.043, green: 0.239, blue: 0.290) // #0B3D4A
    static let deepTide = Color(red: 0.086, green: 0.345, blue: 0.400)
    static let seaGlass = Color(red: 0.494, green: 0.722, blue: 0.659) // #7EB8A8
    static let foam = Color(red: 0.957, green: 0.973, blue: 0.965)
    static let mist = Color(red: 0.890, green: 0.933, blue: 0.922)
    static let ink = Color(red: 0.090, green: 0.145, blue: 0.165)
    static let coralSignal = Color(red: 0.910, green: 0.420, blue: 0.310) // CTA only

    static let brandFont = "Fraunces"
    static let bodyFont = "Manrope"

    static func brand(_ size: CGFloat, weight: Font.Weight = .medium) -> Font {
        .custom(brandFont, size: size).weight(weight)
    }

    static func body(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .custom(bodyFont, size: size).weight(weight)
    }
}

struct CoastalBackground: View {
    var intensity: Double = 1

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    FalaTheme.foam,
                    FalaTheme.mist.opacity(0.95 * intensity),
                    FalaTheme.seaGlass.opacity(0.35 * intensity)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            GeometryReader { geo in
                Canvas { context, size in
                    let cols = 8
                    let rows = 14
                    let w = size.width / CGFloat(cols)
                    let h = size.height / CGFloat(rows)
                    for r in 0..<rows {
                        for c in 0..<cols {
                            let x = CGFloat(c) * w
                            let y = CGFloat(r) * h
                            var path = Path()
                            path.addRoundedRect(
                                in: CGRect(x: x + 4, y: y + 4, width: w - 8, height: h - 8),
                                cornerSize: CGSize(width: 6, height: 6)
                            )
                            let alpha = 0.035 + (Double((r + c) % 3) * 0.012)
                            context.fill(path, with: .color(FalaTheme.atlantic.opacity(alpha * intensity)))
                        }
                    }
                }
                .blur(radius: 0.4)
            }
            .allowsHitTesting(false)
        }
        .ignoresSafeArea()
    }
}

struct WaveMotionModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .offset(y: sin(phase) * 6)
            .onAppear {
                withAnimation(.easeInOut(duration: 3.2).repeatForever(autoreverses: true)) {
                    phase = .pi
                }
            }
    }
}

extension View {
    func waveMotion() -> some View {
        modifier(WaveMotionModifier())
    }
}
