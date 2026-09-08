import SwiftUI

struct RootView: View {
    @EnvironmentObject private var app: AppModel

    var body: some View {
        ZStack {
            CoastalBackground()

            switch app.phase {
            case .home:
                HomeView()
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            case .practice:
                PracticeView()
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            case .progress:
                ProgressScreen()
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.86), value: app.phase)
    }
}
