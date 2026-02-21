#!/bin/bash
set -e

APP_NAME="Qwen3VoiceIME"
BUILD_DIR=".build/release"
APP_DIR="build/${APP_NAME}.app"

# Compile release
swift build -c release --product qwen3-voice-ime

# Create app bundle structure
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Copy files
cp "$BUILD_DIR/qwen3-voice-ime" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp Sources/Qwen3VoiceIME/Info.plist "$APP_DIR/Contents/"

# If you have an icon later
# cp Sources/Qwen3VoiceIME/icon.icns "$APP_DIR/Contents/Resources/AppIcon.icns"

# Codesign
codesign --force --deep --sign - --entitlements Sources/Qwen3VoiceIME/Qwen3VoiceIME.entitlements "$APP_DIR"

echo "Build successful! App created at $APP_DIR"
