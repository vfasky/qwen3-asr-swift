#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Qwen3 Voice IME"
APP_BUNDLE_NAME="${APP_NAME}.app"
DIST_DIR="$ROOT/dist"
BUILD_DIR="$ROOT/.build/release"
APP_DIR="$DIST_DIR/$APP_BUNDLE_NAME"
ICON_SRC="$ROOT/Sources/Qwen3VoiceIME/icon.png"
ICON_NAME="AppIcon"
ICON_FILE="${ICON_NAME}.icns"

WITH_DMG=0
if [ "${1-}" = "--dmg" ]; then
  WITH_DMG=1
fi

mkdir -p "$DIST_DIR"

swift build --disable-sandbox -c release --product qwen3-voice-ime

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

if [ -f "$ICON_SRC" ]; then
  ICONSET_DIR="$BUILD_DIR/${ICON_NAME}.iconset"
  ICON_BASE="icon"
  rm -rf "$ICONSET_DIR"
  mkdir -p "$ICONSET_DIR"
  sips -z 16 16 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_16x16.png" >/dev/null
  sips -z 32 32 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_16x16@2x.png" >/dev/null
  sips -z 32 32 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_32x32.png" >/dev/null
  sips -z 64 64 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_32x32@2x.png" >/dev/null
  sips -z 128 128 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_128x128.png" >/dev/null
  sips -z 256 256 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_128x128@2x.png" >/dev/null
  sips -z 256 256 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_256x256.png" >/dev/null
  sips -z 512 512 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_256x256@2x.png" >/dev/null
  sips -z 512 512 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_512x512.png" >/dev/null
  sips -z 1024 1024 "$ICON_SRC" --out "$ICONSET_DIR/${ICON_BASE}_512x512@2x.png" >/dev/null
  iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/$ICON_FILE"
fi

BUNDLE_ID="${BUNDLE_ID:-com.vfasky.Qwen3VoiceIME}"
VERSION="${VERSION:-0.1.0}"
BUILD_NUMBER="${BUILD_NUMBER:-1}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-Developer ID Application: Guangzhou Yizhi Information Technology Company Limited (92ZWDWJJNB)}"
INSTALLER_IDENTITY="${INSTALLER_IDENTITY:-Developer ID Installer: Guangzhou Yizhi Information Technology Company Limited (92ZWDWJJNB)}"
INSTALL_LOCATION="${INSTALL_LOCATION:-/Applications}"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"
NOTARIZE_TARGET="${NOTARIZE_TARGET:-}"
NOTARY_APPLE_ID="${NOTARY_APPLE_ID:-}"
NOTARY_TEAM_ID="${NOTARY_TEAM_ID:-}"
NOTARY_PASSWORD="${NOTARY_PASSWORD:-}"

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

  <key>CFBundleIconFile</key>
  <string>$ICON_NAME</string>

  <key>LSUIElement</key>
  <string>1</string>

  <key>NSMicrophoneUsageDescription</key>
  <string>This app needs microphone access for speech input.</string>
  <key>NSSpeechRecognitionUsageDescription</key>
  <string>This app needs speech recognition access to transcribe your voice to text.</string>
</dict>
</plist>
EOF

if [ -n "$CODESIGN_IDENTITY" ]; then
  if [ -f "$APP_DIR/Contents/MacOS/mlx.metallib" ]; then
    codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$APP_DIR/Contents/MacOS/mlx.metallib"
  fi
  codesign --force --options runtime --timestamp --entitlements "$ROOT/entitlements.plist" --sign "$CODESIGN_IDENTITY" "$APP_DIR/Contents/MacOS/$APP_NAME"
  codesign --force --deep --options runtime --timestamp --entitlements "$ROOT/entitlements.plist" --sign "$CODESIGN_IDENTITY" "$APP_DIR"
else
  codesign --force --timestamp --entitlements "$ROOT/entitlements.plist" --sign - "$APP_DIR/Contents/MacOS/$APP_NAME"
  codesign --force --deep --timestamp --entitlements "$ROOT/entitlements.plist" --sign - "$APP_DIR"
fi

if [ -n "$INSTALLER_IDENTITY" ]; then
  productbuild \
    --component "$APP_DIR" "$INSTALL_LOCATION" \
    --sign "$INSTALLER_IDENTITY" \
    "$DIST_DIR/qwen3-voice-ime.pkg"
fi

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

if [ -z "$NOTARY_PROFILE" ] && [ -n "$NOTARY_APPLE_ID" ] && [ -n "$NOTARY_TEAM_ID" ] && [ -n "$NOTARY_PASSWORD" ]; then
  NOTARY_PROFILE="qwen3-voice-ime"
  xcrun notarytool store-credentials "$NOTARY_PROFILE" \
    --apple-id "$NOTARY_APPLE_ID" \
    --team-id "$NOTARY_TEAM_ID" \
    --password "$NOTARY_PASSWORD"
fi

if [ -n "$NOTARY_PROFILE" ]; then
  if [ -z "$NOTARIZE_TARGET" ]; then
    if [ "$WITH_DMG" -eq 1 ]; then
      NOTARIZE_TARGET="$DIST_DIR/qwen3-voice-ime.dmg"
    elif [ -n "$INSTALLER_IDENTITY" ]; then
      NOTARIZE_TARGET="$DIST_DIR/qwen3-voice-ime.pkg"
    else
      NOTARIZE_TARGET="$APP_DIR"
    fi
  fi
  xcrun notarytool submit "$NOTARIZE_TARGET" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$NOTARIZE_TARGET"
fi

echo "App bundle created at: $APP_DIR"
if [ "$WITH_DMG" -eq 1 ]; then
  echo "DMG created at: $DIST_DIR/qwen3-voice-ime.dmg"
fi
if [ -n "$INSTALLER_IDENTITY" ]; then
  echo "PKG created at: $DIST_DIR/qwen3-voice-ime.pkg"
fi
if [ -n "$NOTARY_PROFILE" ]; then
  echo "Notarized and stapled: $NOTARIZE_TARGET"
fi
