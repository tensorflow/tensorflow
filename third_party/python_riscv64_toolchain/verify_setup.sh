#!/bin/bash
# Installation Verification Checklist
# Run this script to verify that the RISC-V toolchain is properly set up

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN_DIR="${SCRIPT_DIR}"
TF_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0
warn_count=0

echo "═══════════════════════════════════════════════════════════════════════"
echo "         TensorFlow RISC-V Toolchain Verification Checklist"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Function to check and report
check_item() {
    local description="$1"
    local test_command="$2"
    local level="${3:-error}" # error or warning
    
    echo -n "Checking: $description ... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((pass_count++))
        return 0
    else
        if [ "$level" = "warning" ]; then
            echo -e "${YELLOW}⚠ WARNING${NC}"
            ((warn_count++))
        else
            echo -e "${RED}✗ FAIL${NC}"
            ((fail_count++))
        fi
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CHECKING FILE STRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_item "WORKSPACE file exists" "test -f '${TOOLCHAIN_DIR}/WORKSPACE'"
check_item "BUILD.bazel file exists" "test -f '${TOOLCHAIN_DIR}/BUILD.bazel'"
check_item "riscv64_py_cc_toolchain.bzl exists" "test -f '${TOOLCHAIN_DIR}/riscv64_py_cc_toolchain.bzl'"
check_item "README.md exists" "test -f '${TOOLCHAIN_DIR}/README.md'"
check_item "setup_toolchain.sh exists" "test -f '${TOOLCHAIN_DIR}/setup_toolchain.sh'"
check_item "setup_toolchain.sh is executable" "test -x '${TOOLCHAIN_DIR}/setup_toolchain.sh'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. CHECKING PYTHON CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_item "Python 3.11 available" "command -v python3.11" "warning"
check_item "Python headers directory exists" "test -d '${TOOLCHAIN_DIR}/include'" "warning"
check_item "Python lib directory exists" "test -d '${TOOLCHAIN_DIR}/lib'" "warning"
check_item "Python bin directory exists" "test -d '${TOOLCHAIN_DIR}/bin'" "warning"

# Check for Python.h in various possible locations
if [ -d "${TOOLCHAIN_DIR}/include" ]; then
    check_item "Python.h header file found" "find '${TOOLCHAIN_DIR}/include' -name 'Python.h' | grep -q ." "warning"
fi

# Check for libpython
if [ -d "${TOOLCHAIN_DIR}/lib" ]; then
    check_item "Python shared library found" "ls '${TOOLCHAIN_DIR}/lib'/libpython*.so 2>/dev/null | grep -q ." "warning"
fi

# Check for python symlink
if [ -d "${TOOLCHAIN_DIR}/bin" ]; then
    check_item "Python interpreter link exists" "test -L '${TOOLCHAIN_DIR}/bin/python' || test -f '${TOOLCHAIN_DIR}/bin/python'" "warning"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. CHECKING TENSORFLOW WORKSPACE INTEGRATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_item "TensorFlow WORKSPACE file exists" "test -f '${TF_ROOT}/WORKSPACE'"
check_item "local_repository declaration found" "grep -q 'python_riscv64-unknown-linux-gnu' '${TF_ROOT}/WORKSPACE'"
check_item "register_toolchains declaration found" "grep -q 'python_riscv64_toolchain' '${TF_ROOT}/WORKSPACE'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. CHECKING BAZEL CONFIGURATION (if Bazel is available)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v bazel > /dev/null 2>&1; then
    echo "Bazel found: $(bazel version 2>&1 | head -1)"
    
    # Try to query the toolchain (this might fail if not fully set up)
    cd "${TF_ROOT}"
    check_item "Bazel can see RISC-V repository" "bazel query '@python_riscv64-unknown-linux-gnu//...' --output=label 2>&1 | grep -q 'python_riscv64'" "warning"
else
    echo -e "${YELLOW}⚠ Bazel not found - skipping Bazel checks${NC}"
    ((warn_count++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. CHECKING SYSTEM REQUIREMENTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_item "GCC compiler available" "command -v gcc" "warning"
check_item "Python3 available" "command -v python3" "warning"
check_item "Git available" "command -v git" "warning"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "riscv64" ]; then
    echo -e "Architecture check: ${GREEN}✓ PASS${NC} (running on $ARCH)"
    ((pass_count++))
else
    echo -e "Architecture check: ${YELLOW}⚠ WARNING${NC} (running on $ARCH, not riscv64)"
    ((warn_count++))
    echo "  Note: You may be cross-compiling"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "                          SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Passed:${NC}  $pass_count"
echo -e "${YELLOW}Warnings:${NC} $warn_count"
echo -e "${RED}Failed:${NC}  $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    if [ $warn_count -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Your RISC-V toolchain is properly configured.${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. cd to TensorFlow root: cd ${TF_ROOT}"
        echo "  2. Run: ./configure"
        echo "  3. Build: bazel build --config=opt //tensorflow/tools/pip_package:wheel"
    else
        echo -e "${YELLOW}⚠ Some warnings detected. The toolchain may work, but review warnings above.${NC}"
        echo ""
        echo "Common fixes:"
        echo "  • If Python directories are missing, run: ./setup_toolchain.sh"
        echo "  • If Bazel queries fail, ensure you're in the TensorFlow root"
        echo "  • Architecture warnings are OK if you're setting up for cross-compilation"
    fi
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above before building.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  • Missing files: Ensure all files were created correctly"
    echo "  • Missing Python setup: Run ./setup_toolchain.sh"
    echo "  • WORKSPACE issues: Check that the main WORKSPACE was updated"
    echo ""
    echo "For help, see:"
    echo "  • README.md for complete setup guide"
    echo "  • QUICKSTART.sh for quick reference"
    echo "  • ISSUE_RESOLUTION.md for technical details"
    exit 1
fi
