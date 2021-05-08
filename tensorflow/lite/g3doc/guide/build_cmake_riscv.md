# Cross compilation TensorFlow Lite with CMake for RISC-V platform

This page describes how to build the TensorFlow Lite library for RISC-V
platform.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64).

**Note:** This feature is currently experimental and available since version 2.4
and may change.

### Prerequisites

You need CMake installed and downloaded TensorFlow source code. Please check
[Build TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake)
page for the details.

## Build for RV64 arch

This instruction shows how to build RV64 binary.

#### Download RISC-V clang toolchain and QEMU

These commands install riscv-clang toolchain and QEMU under
${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv folder.

```sh
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv
./riscv_bootstrap.sh
```

**Note1:** If you change the default download path for bootstrap script, you also need to update the
toolchain path setting in riscv-clang.cmake.

**Note2:** The RISC-V clang toolchain is built from https://github.com/llvm/llvm-project. Currently, the toolchain is based on GNU toolchain(including libgcc, GNU linker, and C libraries). You need to build the GNU toolchain first.
```sh
# Get gnu toolchain source code.
git clone https://github.com/riscv/riscv-gnu-toolchain.git

# Set the build and install path
PREFIX=/path/to/the/riscv/clang/install/prefix/you/want

# Build gnu toolchain.
cd riscv-gnu-toolchain
git submodule update --init --recursive
cd riscv-binutils
git fetch origin
git checkout -b rvv-1.0.x-zfh origin/rvv-1.0.x-zfh
cd ..
./configure --with-cmodel=medany --prefix=$PREFIX
make -j linux
```
```sh
# Get LLVM source code.
git clone https://github.com/llvm/llvm-project.git

# Set the build and install path.
PREFIX=/path/to/the/riscv/clang/install/prefix/you/want
BUILDPATH=/path/to/your/build/path
SOURCE=/path/to/your/llvm/source

# Build LLVM/Clang.
mkdir -p $BUILDPATH
cd $BUILDPATH
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD="RISCV" \
      -DLLVM_ENABLE_PROJECTS="clang" \
      -DLLVM_DEFAULT_TARGET_TRIPLE="riscv64-unknown-linux-gnu" \
      -DLLVM_INSTALL_TOOLCHAIN_ONLY=On \
      -DDEFAULT_SYSROOT=../sysroot \
      -G "Ninja" $SOURCE
ninja -j
ninja install
```

**Note3:** The RISC-V QEMU is built from https://github.com/sifive/qemu.git
```sh
# Get QEMU source code.
git clone https://github.com/sifive/qemu.git

# Set the install path.
PREFIX=/path/to/your/qemu/install/prefix/you/want

# Build QEMU.
cd qemu
git checkout origin/v5.2.0-rvv-rvb-zfh -b v5.2.0-rvv-rvb-zfh && git submodule update --init --recursive
mkdir build && cd build
../configure --prefix=$PREFIX --target-list="riscv64-linux-user"
make -j install
```

#### Run CMake

```sh
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../riscv/riscv-clang.cmake \
  ../../..
```

**Note:** XNNPACK is disabled since there is no riscv support in XNNPACK.

#### Run benchmark tool with qemu

```sh
# Please run CMake using previous CMake instructions.
cd ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/build
# Build benchmark tool.
make -j benchmark_model
# Run benchmark tool with qemu.
${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv/Prebuilt/qemu/linux/RISCV/bin/qemu-riscv64 \
  -cpu rv64,x-v=true,x-k=true,vlen=512,elen=64,vext_spec=v1.0 \
  -L ${TENSORFLOW_SRC_PATH}/tensorflow/lite/tools/cmake/riscv/Prebuilt/toolchain/clang/linux/RISCV/sysroot \
  ./tools/benchmark/benchmark_model --help
```
