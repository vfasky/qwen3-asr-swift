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
mkdir -p "$APP_DIR/Contents/Resources/Base.lproj"
mkdir -p "$APP_DIR/Contents/Resources/zh-Hans.lproj"

# Copy files
cp "$BUILD_DIR/qwen3-voice-ime" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp Sources/Qwen3VoiceIME/Info.plist "$APP_DIR/Contents/"

# InfoPlist.strings for Permissions
cat << 'STR' > "$APP_DIR/Contents/Resources/Base.lproj/InfoPlist.strings"
NSMicrophoneUsageDescription = "Need microphone access to record your voice for speech-to-text input.";
NSSpeechRecognitionUsageDescription = "Need speech recognition access to transcribe your voice to text.";
STR

cat << 'STR' > "$APP_DIR/Contents/Resources/zh-Hans.lproj/InfoPlist.strings"
NSMicrophoneUsageDescription = "需要麦克风权限来录制您的语音进行语音识别输入。";
NSSpeechRecognitionUsageDescription = "需要语音识别权限来将您的语音转换为文字。";
STR

# Codesign
codesign --force --deep --sign - --entitlements Sources/Qwen3VoiceIME/Qwen3VoiceIME.entitlements "$APP_DIR"

echo "Build successful! App created at $APP_DIR"
