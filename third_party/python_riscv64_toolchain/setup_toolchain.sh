#!/bin/bash
# Setup script for TensorFlow RISC-V Python toolchain
# This script helps prepare the Python headers and libraries for building TensorFlow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN_DIR="${SCRIPT_DIR}"

echo "===================================="
echo "TensorFlow RISC-V Toolchain Setup"
echo "===================================="
echo ""

# Detect Python version
PYTHON_CMD="${PYTHON_CMD:-python3.11}"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: $PYTHON_CMD not found. Please install Python 3.11 or set PYTHON_CMD environment variable."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Get Python configuration
echo "Detecting Python configuration..."
PYTHON_INCLUDE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBDIR=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LDVERSION=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INTERPRETER=$($PYTHON_CMD -c "import sys; print(sys.executable)")

echo "Python include directory: $PYTHON_INCLUDE"
echo "Python library directory: $PYTHON_LIBDIR"
echo "Python LD version: $PYTHON_LDVERSION"
echo "Python interpreter: $PYTHON_INTERPRETER"
echo ""

# Create directory structure
echo "Creating toolchain directory structure..."
mkdir -p "$TOOLCHAIN_DIR/include"
mkdir -p "$TOOLCHAIN_DIR/lib"
mkdir -p "$TOOLCHAIN_DIR/bin"

# Copy Python headers
echo "Copying Python headers..."
if [ -d "$PYTHON_INCLUDE" ]; then
    cp -r "$PYTHON_INCLUDE" "$TOOLCHAIN_DIR/include/" || {
        echo "Warning: Could not copy headers, trying with sudo..."
        sudo cp -r "$PYTHON_INCLUDE" "$TOOLCHAIN_DIR/include/"
    }
    echo "Headers copied successfully."
else
    echo "Error: Python include directory not found: $PYTHON_INCLUDE"
    exit 1
fi

# Copy or symlink Python library
echo "Setting up Python shared library..."
PYTHON_LIB="libpython${PYTHON_LDVERSION}.so"
if [ -f "$PYTHON_LIBDIR/$PYTHON_LIB" ]; then
    cp "$PYTHON_LIBDIR/$PYTHON_LIB" "$TOOLCHAIN_DIR/lib/" || {
        echo "Warning: Could not copy library, trying with sudo..."
        sudo cp "$PYTHON_LIBDIR/$PYTHON_LIB" "$TOOLCHAIN_DIR/lib/"
    }
    echo "Library copied successfully."
elif [ -f "$PYTHON_LIBDIR/libpython${PYTHON_LDVERSION:0:4}.so" ]; then
    # Try with shortened version (e.g., libpython3.11.so instead of libpython3.11m.so)
    SHORT_VERSION="${PYTHON_LDVERSION:0:4}"
    cp "$PYTHON_LIBDIR/libpython${SHORT_VERSION}.so" "$TOOLCHAIN_DIR/lib/" || {
        echo "Warning: Could not copy library, trying with sudo..."
        sudo cp "$PYTHON_LIBDIR/libpython${SHORT_VERSION}.so" "$TOOLCHAIN_DIR/lib/"
    }
    echo "Library copied successfully."
else
    echo "Error: Python library not found in: $PYTHON_LIBDIR"
    echo "Looking for: $PYTHON_LIB"
    exit 1
fi

# Create symlink to Python interpreter
echo "Creating symlink to Python interpreter..."
ln -sf "$PYTHON_INTERPRETER" "$TOOLCHAIN_DIR/bin/python"
echo "Symlink created."
echo ""

# Verify setup
echo "===================================="
echo "Verifying setup..."
echo "===================================="

# Check for Python.h
if [ -f "$TOOLCHAIN_DIR/include/python${PYTHON_LDVERSION:0:4}/Python.h" ] || \
   [ -f "$TOOLCHAIN_DIR/include/$(basename $PYTHON_INCLUDE)/Python.h" ]; then
    echo "✓ Python headers found"
else
    echo "✗ Python.h not found - check include directory"
    exit 1
fi

# Check for library
if ls "$TOOLCHAIN_DIR/lib/libpython"*.so 1> /dev/null 2>&1; then
    echo "✓ Python library found"
else
    echo "✗ Python library not found - check lib directory"
    exit 1
fi

# Check for interpreter
if [ -L "$TOOLCHAIN_DIR/bin/python" ]; then
    echo "✓ Python interpreter link created"
else
    echo "✗ Python interpreter link not found"
    exit 1
fi

echo ""
echo "===================================="
echo "Setup completed successfully!"
echo "===================================="
echo ""
echo "Directory structure:"
tree -L 2 "$TOOLCHAIN_DIR" 2>/dev/null || find "$TOOLCHAIN_DIR" -maxdepth 2 -type f -o -type l
echo ""
echo "Next steps:"
echo "1. Review and update BUILD.bazel if needed (especially interpreter_path)"
echo "2. Ensure your Python virtual environment is activated"
echo "3. Run: ./configure"
echo "4. Run: bazel build //tensorflow/tools/pip_package:wheel"
echo ""
