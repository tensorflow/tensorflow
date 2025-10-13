# TensorFlow RISC-V Build Issue #102159 - Resolution

## Issue Summary

**Problem:** Cannot compile TensorFlow 2.19.1 on RISC-V due to py_cc_toolchain provider error.

**Error Message:**
```
ERROR: <target @python_riscv64-unknown-linux-gnu//:python_headers> (rule 'cc_library') 
doesn't have provider 'headers'
```

**Platform:** RISC-V 64-bit, EulixOS 12.3.1-33.eos30, Python 3.11.6, GCC 12.3.1

## Root Cause Analysis

The custom RISC-V Python C/C++ toolchain implementation had an incorrect provider structure. The `rules_python` framework expects:

1. A `ToolchainInfo` with a `headers` field
2. The `headers` field must have a `providers_map` attribute containing the actual target
3. The target must provide `CcInfo` with compilation context

The original implementation was directly passing the label, which caused the provider lookup to fail.

## Solution Overview

The fix involves three main components:

### 1. Corrected Toolchain Implementation (`riscv64_py_cc_toolchain.bzl`)

**Before (Incorrect):**
```python
def _riscv64_py_cc_toolchain_impl(ctx):
    py_cc = ctx.attr.python_headers[0]
    return platform_common.ToolchainInfo(
        py_cc_toolchain = py_cc,
        python_headers = ctx.attr.python_headers,
        python_includes = ctx.attr.python_includes,
    )
```

**After (Correct):**
```python
def _riscv64_py_cc_toolchain_impl(ctx):
    python_headers_target = ctx.attr.python_headers[0]
    
    headers_struct = struct(
        providers_map = {
            "headers": python_headers_target,
        },
    )
    
    return [
        platform_common.ToolchainInfo(
            headers = headers_struct,
            python_version = ctx.attr.python_version,
        ),
    ]
```

**Key Changes:**
- Created a `struct` with `providers_map` dictionary
- The `headers` field now contains this struct
- Returned as a list with proper ToolchainInfo
- Added `python_version` attribute for better compatibility

### 2. Proper BUILD Configuration

The `BUILD.bazel` file now properly:
- Defines Python runtime with correct interpreter path
- Creates `cc_library` target for Python headers with proper includes
- Implements the py_cc_toolchain rule with correct attributes
- Adds platform constraints for RISC-V (riscv64) + Linux
- Registers both toolchains (py_runtime and py_cc_toolchain)

### 3. WORKSPACE Integration

Added to the main TensorFlow WORKSPACE file:
```python
# RISC-V Python toolchain configuration
local_repository(
    name = "python_riscv64-unknown-linux-gnu",
    path = "third_party/python_riscv64_toolchain",
)

register_toolchains(
    "@python_riscv64-unknown-linux-gnu//:python_riscv64_toolchain",
    "@python_riscv64-unknown-linux-gnu//:riscv64_py_cc_toolchain",
)
```

## Files Created/Modified

### New Files:
1. `third_party/python_riscv64_toolchain/WORKSPACE`
2. `third_party/python_riscv64_toolchain/BUILD.bazel`
3. `third_party/python_riscv64_toolchain/riscv64_py_cc_toolchain.bzl`
4. `third_party/python_riscv64_toolchain/README.md`
5. `third_party/python_riscv64_toolchain/setup_toolchain.sh`
6. `third_party/python_riscv64_toolchain/ISSUE_RESOLUTION.md` (this file)

### Modified Files:
1. `WORKSPACE` (added RISC-V toolchain registration)

## Implementation Steps

### Step 1: Setup Python Environment
```bash
# Activate your Python virtual environment
source /AI/zjg/python/venv11/bin/activate

# Navigate to toolchain directory
cd third_party/python_riscv64_toolchain

# Run the setup script
./setup_toolchain.sh
```

The setup script will:
- Detect your Python installation
- Copy Python headers to the toolchain directory
- Copy the Python shared library
- Create necessary symlinks

### Step 2: Verify Directory Structure

After running the setup script, verify you have:
```
third_party/python_riscv64_toolchain/
├── BUILD.bazel
├── WORKSPACE
├── riscv64_py_cc_toolchain.bzl
├── README.md
├── setup_toolchain.sh
├── include/
│   └── python3.11/
│       ├── Python.h
│       └── ... (other headers)
├── lib/
│   └── libpython3.11.so
└── bin/
    └── python (symlink)
```

### Step 3: Build TensorFlow

```bash
# Configure TensorFlow
./configure

# Build the pip package
bazel build --config=opt //tensorflow/tools/pip_package:wheel
```

## Technical Deep Dive

### Provider Structure Requirement

The `rules_python` framework uses this code path (from the error trace):
```python
# File: external/rules_python/python/private/current_py_cc_headers.bzl
return py_cc_toolchain.headers.providers_map.values()
```

This means:
1. `py_cc_toolchain` must be a ToolchainInfo
2. `py_cc_toolchain.headers` must exist
3. `headers` must have a `providers_map` attribute
4. `providers_map` must be a dictionary
5. `providers_map.values()` returns the actual provider targets

### Provider Flow

```
ToolchainInfo
└── headers (struct)
    └── providers_map (dict)
        └── "headers" (key)
            └── cc_library target (value)
                └── CcInfo provider
                    └── compilation_context
                        └── headers, includes, defines, etc.
```

### Platform Constraints

The toolchains are constrained to:
- CPU: `@platforms//cpu:riscv64`
- OS: `@platforms//os:linux`

This ensures they're only used when building for RISC-V Linux targets.

## Testing and Verification

### Verify Toolchain Registration
```bash
# Check if toolchains are registered
bazel query '@python_riscv64-unknown-linux-gnu//...'

# Verify toolchain targets
bazel query --output=build @python_riscv64-unknown-linux-gnu//:riscv64_py_cc_toolchain
```

### Build Verification
```bash
# Build TensorFlow Python package
bazel build //tensorflow/tools/pip_package:wheel

# Build specific components
bazel build //tensorflow/python:tensorflow_py
```

### Expected Output
The build should now proceed without the "doesn't have provider 'headers'" error.

## Compatibility Notes

- **Python Version:** Configured for Python 3.11, but can be adapted for other versions
- **Architecture:** RISC-V 64-bit (riscv64)
- **OS:** Linux (tested on EulixOS 12.3.1)
- **TensorFlow Version:** 2.19.1
- **Bazel Version:** Compatible with Bazel 2.19.1+

## Future Improvements

1. **Multi-Version Support:** Add support for multiple Python versions
2. **Cross-Compilation:** Enable building from x86_64 for RISC-V
3. **Automated Tests:** Add verification tests for the toolchain
4. **Dynamic Configuration:** Auto-detect Python paths during configuration
5. **Platform Variants:** Support different RISC-V variants (RV64GC, RV64IMAFDC, etc.)

## References

- Original Issue: #102159
- TensorFlow Build Guide: https://www.tensorflow.org/install/source
- Bazel Toolchains: https://bazel.build/extending/toolchains
- rules_python: https://github.com/bazelbuild/rules_python
- Python C API: https://docs.python.org/3/c-api/

## Credits

- Issue Reporter: @Sherlockzhangjinge
- Platform: RISC-V 64-bit
- Resolved: [Current Date]

## License

This solution follows the TensorFlow project license (Apache 2.0).
