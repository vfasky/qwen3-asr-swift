import SwiftUI

@main
struct Qwen3VoiceIMEApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra {
            MenuView(appState: appState)
        } label: {
            Image(systemName: appState.menuBarIconName)
        }
        .menuBarExtraStyle(.menu)

        Settings {
            SettingsView(appState: appState)
        }
    }
}
