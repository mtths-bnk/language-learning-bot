import Foundation

struct ConversationTurn: Identifiable, Equatable {
    enum Role: String {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    let text: String
    let analysis: UtteranceAnalysis?
    let createdAt: Date

    init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        analysis: UtteranceAnalysis? = nil,
        createdAt: Date = .now
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.analysis = analysis
        self.createdAt = createdAt
    }
}
