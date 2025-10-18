#!/bin/bash
# Quick Start Guide for TensorFlow RISC-V Build
# This script provides copy-paste commands for rapid setup

cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║                   TensorFlow RISC-V Quick Start Guide                      ║
╚════════════════════════════════════════════════════════════════════════════╝

This guide helps you quickly set up and build TensorFlow on RISC-V.

STEP 1: Navigate to the toolchain directory
────────────────────────────────────────────────────────────────────────────

cd third_party/python_riscv64_toolchain


STEP 2: Run the setup script
────────────────────────────────────────────────────────────────────────────

./setup_toolchain.sh

This will automatically:
  ✓ Detect your Python installation
  ✓ Copy Python headers and libraries
  ✓ Set up the directory structure


STEP 3: Review and update BUILD.bazel (if needed)
────────────────────────────────────────────────────────────────────────────

Check the interpreter_path in BUILD.bazel:
  • Default: /AI/zjg/python/venv11/bin/python
  • Update if your Python is in a different location

# To edit:
nano BUILD.bazel  # or use your preferred editor


STEP 4: Return to TensorFlow root and configure
────────────────────────────────────────────────────────────────────────────

cd ../..
./configure


STEP 5: Build TensorFlow
────────────────────────────────────────────────────────────────────────────

# Build the pip package
bazel build --config=opt //tensorflow/tools/pip_package:wheel

# Or build with specific optimizations
bazel build --config=opt --copt=-march=rv64gc //tensorflow/tools/pip_package:wheel


STEP 6: Install the built wheel
────────────────────────────────────────────────────────────────────────────

# Activate your virtual environment (if not already activated)
source /AI/zjg/python/venv11/bin/activate

# Install the wheel
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-*.whl


═══════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING
───────────────────────────────────────────────────────────────────────────

Problem: "setup_toolchain.sh: Permission denied"
Solution: chmod +x setup_toolchain.sh

Problem: "Python.h not found" during build
Solution: Ensure setup_toolchain.sh completed successfully
         Check that headers are in include/python3.11/

Problem: "libpython3.11.so not found"
Solution: Verify the library is in lib/ directory
         Run: ls -l lib/

Problem: "Toolchain not found"
Solution: Ensure WORKSPACE file has the toolchain registration
         The main TensorFlow WORKSPACE should contain:
           local_repository(name = "python_riscv64-unknown-linux-gnu", ...)
           register_toolchains("@python_riscv64-unknown-linux-gnu//...")

───────────────────────────────────────────────────────────────────────────

VERIFICATION COMMANDS
───────────────────────────────────────────────────────────────────────────

# Check toolchain registration
bazel query '@python_riscv64-unknown-linux-gnu//...'

# Verify Python headers
ls -la third_party/python_riscv64_toolchain/include/python3.11/Python.h

# Verify Python library
ls -la third_party/python_riscv64_toolchain/lib/libpython*.so

# Test simple build
bazel build //tensorflow:tensorflow

───────────────────────────────────────────────────────────────────────────

IMPORTANT NOTES
───────────────────────────────────────────────────────────────────────────

1. Ensure you're on a RISC-V system or using proper cross-compilation
2. The build may take several hours depending on your system
3. You need at least 32GB RAM (or sufficient swap space)
4. Monitor disk space - build artifacts can exceed 50GB

───────────────────────────────────────────────────────────────────────────

For detailed documentation, see:
  • README.md - Complete setup guide
  • ISSUE_RESOLUTION.md - Technical details about the fix

═══════════════════════════════════════════════════════════════════════════

EOF
