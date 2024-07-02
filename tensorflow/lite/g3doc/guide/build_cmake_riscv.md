# Cross compilation TensorFlow Lite with CMake for RISC-V platform

This page describes how to build the TensorFlow Lite library for RISC-V
platform.

The following instructions have been tested on Ubuntu 20.04 64-bit PC.

**Note:** This feature is available since version 2.14.

### Prerequisites

You need CMake installed and downloaded TensorFlow source code. Please check
[Build TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake)
page for the details.

## Build for RV64 arch

These instructions show how to download, build, and run RV64 binary.

### Download RISC-V toolchain and QEMU

RISC-V prebuilt toolchain(GCC & LLVM) and qemu have been provided in release page of [riscv-collab/riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain/releases).

```sh
# Example: Download 2023.07.07 nightly GCC
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.11.20/riscv64-glibc-ubuntu-22.04-gcc-nightly-2023.11.20-nightly.tar.gz
tar zxvf riscv64-glibc-ubuntu-22.04-gcc-nightly-2023.11.20-nightly.tar.gz
```
- gcc is in `riscv/bin/riscv64-unknown-linux-gnu-gcc`.
- g++ is in `riscv/bin/riscv64-unknown-linux-gnu-g++`.
- User mode QEMU is in `riscv/bin/qemu-riscv64`.

**Note:** If you download LLVM toolchain,
- clang is in `riscv/bin/riscv64-unknown-linux-gnu-clang`.
- clang++ is in `riscv/bin/riscv64-unknown-linux-gnu-clang++`.


### Run CMake

```sh
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake
mkdir build
cd build
RVCC_PREFIX=${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv/bin/riscv64-unknown-linux-gnu-
RVCC_FLAGS="-march=rv64gcv"
cmake -DCMAKE_C_COMPILER=${RVCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${RVCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${RVCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${RVCC_FLAGS}" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  ../../..
```

**Note:** XNNPACK is enabled by default in TFLite and the RISC-V vector optimized kernels are already supported in XNNPACK.

### Run benchmark tool with qemu

```sh
# Please run CMake using previous CMake instructions.
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/build
# Build benchmark tool.
make -j benchmark_model
# Run benchmark tool with qemu.
${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv/bin/qemu-riscv64 \
  -cpu rv64,zba=true,zbb=true,zbc=true,zbs=true,v=true,vlen=512,elen=64,vext_spec=v1.0 \
  -L ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv/sysroot \
  ./tools/benchmark/benchmark_model --help
```
