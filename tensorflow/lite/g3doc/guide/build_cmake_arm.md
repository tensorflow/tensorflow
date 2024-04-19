# Cross compilation TensorFlow Lite with CMake

This page describes how to build the TensorFlow Lite library for various ARM
devices.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, TensorFlow devel docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This feature is available since version 2.4.

### Prerequisites

You need CMake installed and downloaded TensorFlow source code. Please check
[Build TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake)
page for the details.

### Check your target environment

The following examples are tested under Raspberry Pi OS, Ubuntu Server 20.04 LTS
and Mendel Linux 4.0. Depending on your target glibc version and CPU
capabilities, you may need to use different version of toolchain and build
parameters.

#### Checking glibc version

```sh
ldd --version
```

<pre  class="tfo-notebook-code-cell-output">
ldd (Debian GLIBC 2.28-10) 2.28
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
</pre>

#### Checking ABI compatibility

If your target is ARM 32-bit, there are two ABI available depending on VFP
availity. [armhf](https://wiki.debian.org/ArmHardFloatPort) and
[armel](https://wiki.debian.org/ArmEabiPort). This document shows an armhf
example, you need to use different toolchain for armel targets.

#### Checking CPU capability

For ARMv7, you should know target's supported VFP version and NEON availability.

```sh
cat /proc/cpuinfo
```

<pre  class="tfo-notebook-code-cell-output">
processor   : 0
model name  : ARMv7 Processor rev 3 (v7l)
BogoMIPS    : 108.00
Features    : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part    : 0xd08
CPU revision    : 3
</pre>

## Build for AArch64 (ARM64)

This instruction shows how to build AArch64 binary which is compatible with
[Coral Mendel Linux 4.0](https://coral.ai/), Raspberry Pi (with
[Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi)
installed).

#### Download toolchain

These commands install `gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu` toolchain
under ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**Note:** Binaries built with GCC 8.3 require glibc 2.28 or higher. If your
target has lower glibc version, you need to use older GCC toolchain.

#### Run CMake

```sh
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-
ARMCC_FLAGS="-funsafe-math-optimizations"
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  ../tensorflow/lite/
```

**Note:** You can enable GPU delegate with `-DTFLITE_ENABLE_GPU=ON` if your
target device supports OpenCL 1.2 or higher.

## Build for ARMv7 NEON enabled

This instruction shows how to build ARMv7 with VFPv4 and NEON enabled binary
which is compatible with Raspberry Pi 3 and 4.

#### Download toolchain

These commands install `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf`
toolchain under ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Note:** Binaries built with GCC 8.3 require glibc 2.28 or higher. If your
target has lower glibc version, you need to use older GCC toolchain.

#### Run CMake

```sh
ARMCC_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -funsafe-math-optimizations -mfp16-format=ieee"
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  ../tensorflow/lite/
```

**Note:** Since ARMv7 architecture is diverse, you may need to update
`ARMCC_FLAGS` for your target device profiles. For example, when compiling with
XNNPACK enabled (i.e. `XNNPACK=ON`) in Tensorflow Lite 2.8, please add
`-mfp16-format=ieee` to `ARMCC_FLAGS`.

## Build for Raspberry Pi Zero (ARMv6)

This instruction shows how to build ARMv6 binary which is compatible with
Raspberry Pi Zero.

#### Download toolchain

These commands install `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf`
toolchain under ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Note:** Binaries built with GCC 8.3 require glibc 2.28 or higher. If your
target has lower glibc version, you need to use older GCC toolchain.

#### Run CMake

```sh
ARMCC_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard -funsafe-math-optimizations"
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv6 \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  ../tensorflow/lite/
```

**Note:** XNNPACK is disabled since there is no NEON support.
