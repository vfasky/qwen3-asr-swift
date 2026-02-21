#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Qwen3 Voice IME"
APP_BUNDLE_NAME="${APP_NAME}.app"
DIST_DIR="$ROOT/dist"
BUILD_DIR="$ROOT/.build/release"
APP_DIR="$DIST_DIR/$APP_BUNDLE_NAME"

WITH_DMG=0
if [ "${1-}" = "--dmg" ]; then
  WITH_DMG=1
fi

mkdir -p "$DIST_DIR"

swift build -c release --product qwen3-voice-ime

# Build MLX Metal library (required for GPU kernels) if missing
if [ ! -f "$BUILD_DIR/mlx.metallib" ]; then
  "$ROOT/scripts/build_mlx_metallib.sh" release
fi

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp "$BUILD_DIR/qwen3-voice-ime" "$APP_DIR/Contents/MacOS/$APP_NAME"

# Copy mlx.metallib next to the executable so MLX can find it
if [ -f "$BUILD_DIR/mlx.metallib" ]; then
  cp "$BUILD_DIR/mlx.metallib" "$APP_DIR/Contents/MacOS/mlx.metallib"
else
  echo "warning: $BUILD_DIR/mlx.metallib not found; MLX GPU kernels may fail to load" >&2
fi

BUNDLE_ID="${BUNDLE_ID:-com.example.Qwen3VoiceIME}"
VERSION="${VERSION:-0.1.0}"
BUILD_NUMBER="${BUILD_NUMBER:-1}"

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>

  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>

  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>

  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>

  <key>CFBundlePackageType</key>
  <string>APPL</string>

  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>

  <key>CFBundleVersion</key>
  <string>$BUILD_NUMBER</string>

  <key>LSUIElement</key>
  <string>1</string>

  <key>NSMicrophoneUsageDescription</key>
  <string>This app needs microphone access for speech input.</string>
</dict>
</plist>
EOF

if [ "$WITH_DMG" -eq 1 ]; then
  DMG_DIR="$DIST_DIR/Qwen3-Voice-IME"
  rm -rf "$DMG_DIR"
  mkdir -p "$DMG_DIR"
  cp -R "$APP_DIR" "$DMG_DIR/"
  hdiutil create \
    -volname "Qwen3 Voice IME" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    "$DIST_DIR/qwen3-voice-ime.dmg"
fi

echo "App bundle created at: $APP_DIR"
if [ "$WITH_DMG" -eq 1 ]; then
  echo "DMG created at: $DIST_DIR/qwen3-voice-ime.dmg"
fi
