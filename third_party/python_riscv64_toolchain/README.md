# TensorFlow RISC-V Build Configuration

This document describes how to build TensorFlow 2.19.1 on RISC-V 64-bit architecture.

## Overview

This configuration adds support for building TensorFlow on RISC-V (riscv64) platforms using custom Python and C/C++ toolchains.

## Prerequisites

1. **RISC-V System Requirements:**
   - RISC-V 64-bit processor (riscv64)
   - Linux operating system (tested on EulixOS 12.3.1-33.eos30)
   - GCC 12.3.1 or later

2. **Python Requirements:**
   - Python 3.11.6 (or compatible version)
   - Python development headers and libraries
   - Python virtual environment (recommended)

3. **Build Tools:**
   - Bazel 2.19.1
   - Git

## Setup Instructions

### Step 1: Prepare Python Environment

1. Create a Python virtual environment:
   ```bash
   python3.11 -m venv /AI/zjg/python/venv11
   source /AI/zjg/python/venv11/bin/activate
   ```

2. Install required Python packages:
   ```bash
   pip install numpy wheel
   pip install keras_preprocessing --no-deps
   ```

### Step 2: Organize Python Headers and Libraries

The toolchain expects Python headers and libraries to be organized in the following structure under `third_party/python_riscv64_toolchain/`:

```
third_party/python_riscv64_toolchain/
├── BUILD.bazel
├── WORKSPACE
├── riscv64_py_cc_toolchain.bzl
├── include/
│   └── python3.11/
│       ├── Python.h
│       ├── pyconfig.h
│       └── ... (all other Python headers)
├── lib/
│   └── libpython3.11.so
└── bin/
    └── python -> /AI/zjg/python/venv11/bin/python
```

**To set this up:**

```bash
cd third_party/python_riscv64_toolchain

# Create directories
mkdir -p include lib bin

# Copy Python headers
# Find your Python include directory first:
python3.11 -c "import sysconfig; print(sysconfig.get_path('include'))"
# Example output: /usr/include/python3.11

# Copy the headers
cp -r /usr/include/python3.11 include/

# Copy or link the Python shared library
# Find your Python library directory:
python3.11 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
# Example output: /usr/lib64

# Copy the library
cp /usr/lib64/libpython3.11.so lib/

# Create symbolic link to your Python interpreter
ln -s /AI/zjg/python/venv11/bin/python bin/python
```

### Step 3: Update Configuration Paths (if needed)

If your Python installation is in a different location, update the following files:

**In `BUILD.bazel`:**
- Update `interpreter_path` in the `py_runtime` rule (line 18)
- Update `includes` paths if using a different Python version (line 59-61)

**In `riscv64_py_cc_toolchain.bzl`:**
- Update `python_version` attribute if using a different Python version (line 50)

### Step 4: Build TensorFlow

Now you can build TensorFlow with RISC-V support:

```bash
# Configure the build
./configure

# Build the pip package
bazel build --config=opt --config=riscv64 //tensorflow/tools/pip_package:wheel

# Or build specific targets
bazel build --config=opt //tensorflow:tensorflow
```

## Key Changes and Fixes

### Problem
The original error occurred because:
```
Error: <target @python_riscv64-unknown-linux-gnu//:python_headers> (rule 'cc_library') 
doesn't have provider 'headers'
```

### Root Cause
The `py_cc_toolchain` implementation was not providing the correct provider structure. The `rules_python` framework expects:
- A `headers` field in the ToolchainInfo
- The `headers` field must have a `providers_map` attribute
- The `providers_map` must contain the actual target providing CcInfo

### Solution
The fixed `riscv64_py_cc_toolchain.bzl` now properly structures the provider:

```python
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

## Toolchain Components

### 1. Python Runtime Toolchain (`python_riscv64_toolchain`)
- Provides the Python interpreter for running Python code during the build
- Type: `@rules_python//python:toolchain_type`

### 2. Python C/C++ Toolchain (`riscv64_py_cc_toolchain`)
- Provides Python C API headers for compiling native extensions
- Type: `@rules_python//python/cc:toolchain_type`

Both toolchains are registered with platform constraints:
- `@platforms//cpu:riscv64`
- `@platforms//os:linux`

## Build Configuration

To use these toolchains automatically when building for RISC-V, you can create a `.bazelrc` entry:

```bash
# Add to your .bazelrc or use --config=riscv64
build:riscv64 --platforms=@platforms//os:linux
build:riscv64 --cpu=riscv64
build:riscv64 --host_cpu=riscv64
```

## Troubleshooting

### Issue: "Cannot find Python headers"
**Solution:** Ensure headers are copied to the correct location and the glob pattern in BUILD.bazel matches your directory structure.

### Issue: "libpython3.11.so not found"
**Solution:** Verify the library is in the `lib/` directory and has the correct name for your Python version.

### Issue: "Platform mismatch"
**Solution:** Ensure you're building on a RISC-V system or using proper cross-compilation settings.

### Issue: "Toolchain not found"
**Solution:** Verify the WORKSPACE file has been updated with the local_repository and register_toolchains calls.

## Verification

To verify the toolchains are properly registered:

```bash
bazel query --output=build @python_riscv64-unknown-linux-gnu//:riscv64_py_cc_toolchain
bazel query --output=build @python_riscv64-unknown-linux-gnu//:python_riscv64_toolchain
```

## Additional Resources

- [Bazel Toolchains Documentation](https://bazel.build/extending/toolchains)
- [rules_python Documentation](https://github.com/bazelbuild/rules_python)
- [TensorFlow Build from Source](https://www.tensorflow.org/install/source)

## Contributing

If you encounter issues or have improvements for RISC-V support, please:
1. Check existing issues on the TensorFlow GitHub repository
2. Create a detailed issue report with your configuration
3. Consider contributing fixes via pull requests

## License

This configuration follows the same license as TensorFlow (Apache 2.0).
