# TensorFlow CPU No-AVX Builder

Dockerized hermetic environments and automated scripts to build non-AVX CPU
TensorFlow pip wheels from source on Linux and Windows hosts.

## Overview

Compiles TensorFlow targeting the Intel **Westmere** architecture
(`-march=westmere -Wno-sign-compare`). Westmere supports instruction sets up to
**SSE4.2** and **AES-NI**, but pre-dates **AVX**, ensuring full compatibility
with older CPUs, hypervisors, and virtualized environments without AVX support.

## Files

-   `Dockerfile`: Ubuntu 22.04 base container with Clang-17, Python 3.11,
    Bazelisk, and required build toolchains.
-   `build_tf_noavx.sh`: In-container configuration, build, and packaging
    script.
-   `run_build.sh`: Linux host helper script to run hermetic Docker builds.
-   `run_build.bat`: Windows host helper script to run hermetic Docker builds.
-   `../windows/build_tf_windows_noavx.bat`: Native Windows batch build script
    (MSVC/Clang, without Docker).
-   `../windows/build_tf_windows_noavx.sh`: Native Windows MSYS2/Git Bash build
    script.
-   `../windows/test_tf_windows_noavx.sh`: Python runtime verification script
    for Windows wheels.

## Usage

### 1. Windows (Hermetic Docker Build)

From Windows Command Prompt (`cmd.exe`) or PowerShell with Docker Desktop
running:

```bat
tensorflow\tools\ci_build\noavx\run_build.bat
```

This will:

1.  Build the hermetic `tf-cpu-noavx-builder` Docker image.
2.  Mount the local repository, sanitize build scripts for Unix LF line endings,
    and run the Bazel build.
3.  Automatically cache compilation artifacts in `.bazel_cache\` for fast
    incremental rebuilds.
4.  Run in-container Python verification tests to validate tensor math and
    confirm that No-AVX flags are honored.
5.  Export the final `.whl` package to `build_output\`.

### 2. Linux (Hermetic Docker Build)

From the root of the TensorFlow repository on a Linux host with Docker
installed:

```bash
./tensorflow/tools/ci_build/noavx/run_build.sh
```

Or run manually:

```bash
docker build -t tf-cpu-noavx-builder tensorflow/tools/ci_build/noavx
docker run --rm \
  --name tf-cpu-noavx-build-run \
  -v "$(pwd):/tensorflow" \
  -v "$(pwd)/build_output:/tf_wheel" \
  -v "$HOME/.cache/bazel:/root/.cache" \
  tf-cpu-noavx-builder
```

The output wheel will be placed in `build_output/`.

### 3. Windows Native Build (Without Docker)

To build natively on a Windows machine configured with Clang and Bazelisk:

From Windows Command Prompt (`cmd.exe`):

```bat
tensorflow\tools\ci_build\windows\build_tf_windows_noavx.bat
```

Or from MSYS2 / Git Bash:

```bash
./tensorflow/tools/ci_build/windows/build_tf_windows_noavx.sh
./tensorflow/tools/ci_build/windows/test_tf_windows_noavx.sh
```

The resulting `.whl` package will be placed in `build_output/`.
