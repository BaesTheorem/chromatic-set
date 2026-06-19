#!/bin/bash
# Build the Chromatic Set macOS launcher into ~/Desktop/Apps/Chromatic Set.app
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Chromatic Set"
APPS_DIR="$HOME/Desktop/Apps"
APP_PATH="$APPS_DIR/${APP_NAME}.app"
PORT=5018

mkdir -p "$APPS_DIR"

# 1. venv + deps
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
  python3 -m venv "$SCRIPT_DIR/.venv"
fi
"$SCRIPT_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$SCRIPT_DIR/.venv/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# 2. icon
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/create_icon.py" || echo "icon step skipped"

# 3. bundle
rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
[ -f "$SCRIPT_DIR/AppIcon.icns" ] && cp "$SCRIPT_DIR/AppIcon.icns" "$APP_PATH/Contents/Resources/"

cat > "$APP_PATH/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Chromatic Set</string>
    <key>CFBundleIdentifier</key><string>com.exobrain.chromatic-set</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>launch</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key><false/>
</dict>
</plist>
PLIST

cat > "$APP_PATH/Contents/MacOS/launch" << LAUNCHER
#!/bin/bash
DIR="$SCRIPT_DIR"
PORT=$PORT
# start server if not already up
if ! curl -s -o /dev/null "http://localhost:\$PORT/"; then
  "\$DIR/.venv/bin/python" "\$DIR/app.py" >/tmp/chromatic-set.log 2>&1 &
  for i in \$(seq 1 30); do
    curl -s -o /dev/null "http://localhost:\$PORT/" && break
    sleep 0.3
  done
fi
open "http://localhost:\$PORT/"
LAUNCHER
chmod +x "$APP_PATH/Contents/MacOS/launch"

echo "Built $APP_PATH"
