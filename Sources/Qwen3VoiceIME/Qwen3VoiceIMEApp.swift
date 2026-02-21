import SwiftUI

@main
struct Qwen3VoiceIMEApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra {
            MenuView(appState: appState)
        } label: {
            Label("Qwen3 Voice IME", systemImage: appState.menuBarIconName)
        }
        .menuBarExtraStyle(.menu)

        Settings {
            SettingsView(appState: appState)
        }
    }
}

