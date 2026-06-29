#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BLS Meta-Analysis Pipeline — Mac Build Script
# Run this on your Mac:  bash build_mac.sh
# Output: dist/BLS_Meta_Pipeline.app
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=========================================="
echo " BLS Meta Pipeline — Mac Build"
echo "=========================================="

# ── 1. Check Python (3.9+ required) ──────────────────────────────────────────
PYTHON=$(which python3 || which python)
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON ($PY_VER)"
if [[ $(echo "$PY_VER < 3.9" | bc) -eq 1 ]]; then
    echo "ERROR: Python 3.9+ required. Install from python.org or via homebrew: brew install python@3.12"
    exit 1
fi

# ── 2. Create/activate virtualenv ────────────────────────────────────────────
VENV_DIR="venv_mac"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Virtualenv: $VENV_DIR"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "Installing dependencies (this may take a few minutes on first run)..."
pip install --quiet --upgrade pip

pip install --quiet \
    pyinstaller \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    statsmodels \
    matplotlib \
    pillow \
    pyyaml \
    openpyxl

echo "Dependencies installed."

# ── 4. Run PyInstaller ────────────────────────────────────────────────────────
echo ""
echo "Building .app bundle..."
pyinstaller BLS_Meta_Pipeline_mac.spec --noconfirm

# ── 5. Verify output ──────────────────────────────────────────────────────────
APP="dist/BLS_Meta_Pipeline.app"
if [ -d "$APP" ]; then
    SIZE=$(du -sh "$APP" | cut -f1)
    echo ""
    echo "=========================================="
    echo " Build successful!"
    echo " Output: $(pwd)/$APP"
    echo " Size:   $SIZE"
    echo "=========================================="
    echo ""
    echo "To distribute: zip the .app bundle"
    echo "  zip -r BLS_Meta_Pipeline_mac.zip dist/BLS_Meta_Pipeline.app"
    echo ""
    echo "Note: first launch on other Macs may show a Gatekeeper warning."
    echo "Right-click the .app → Open to bypass, or run:"
    echo "  xattr -dr com.apple.quarantine dist/BLS_Meta_Pipeline.app"
else
    echo "ERROR: Build failed — .app not found"
    exit 1
fi
