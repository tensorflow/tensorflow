#!/bin/bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Download GCC 8.3 based toolchains.
# Using up-to-date toolchain introduces compatibility issues.
# https://github.com/tensorflow/tensorflow/issues/59631
#
# In Bazel build, we uses GCC 11.3 based toolchains to support FP16 better
# with XNNPACK. https://github.com/tensorflow/tensorflow/pull/57585

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."

TOOLCHAINS_DIR=$(realpath tensorflow/lite/tools/cmake/toolchains)
mkdir -p ${TOOLCHAINS_DIR}

case $1 in
  armhf_vfpv3)
    if [[ ! -d "${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf" ]]; then
      curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz >&2
      tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${TOOLCHAINS_DIR} >&2
    fi
    ARMCC_ROOT=${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
    echo "ARMCC_FLAGS=\"-march=armv7-a -mfpu=neon-vfpv3 -funsafe-math-optimizations \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include-fixed \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/include/c++/8.3.0 \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/libc/usr/include \
      -isystem \"\${CROSSTOOL_PYTHON_INCLUDE_PATH}\" \
      -isystem /usr/include\""
    echo "ARMCC_PREFIX=${ARMCC_ROOT}/bin/arm-linux-gnueabihf-"
    ;;
  armhf)
    if [[ ! -d "${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf" ]]; then
      curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz >&2
      tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${TOOLCHAINS_DIR} >&2
    fi
    ARMCC_ROOT=${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
    echo "ARMCC_FLAGS=\"-march=armv7-a -mfpu=neon-vfpv4 -funsafe-math-optimizations \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include-fixed \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/include/c++/8.3.0 \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/libc/usr/include \
      -isystem \"\${CROSSTOOL_PYTHON_INCLUDE_PATH}\" \
      -isystem /usr/include\""
    echo "ARMCC_PREFIX=${ARMCC_ROOT}/bin/arm-linux-gnueabihf-"
    ;;
  aarch64)
    if [[ ! -d "${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu" ]]; then
      curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz >&2
      tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${TOOLCHAINS_DIR} >&2
    fi
    ARMCC_ROOT=${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu
    echo "ARMCC_FLAGS=\"-funsafe-math-optimizations \
      -isystem ${ARMCC_ROOT}/lib/gcc/aarch64-linux-gnu/8.3.0/include \
      -isystem ${ARMCC_ROOT}/lib/gcc/aarch64-linux-gnu/8.3.0/include-fixed \
      -isystem ${ARMCC_ROOT}/aarch64-linux-gnu/include/c++/8.3.0 \
      -isystem ${ARMCC_ROOT}/aarch64-linux-gnu/libc/usr/include \
      -isystem \"\${CROSSTOOL_PYTHON_INCLUDE_PATH}\" \
      -isystem /usr/include\""
    echo "ARMCC_PREFIX=${ARMCC_ROOT}/bin/aarch64-linux-gnu-"
    ;;
  rpi0)
    if [[ ! -d "${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf" ]]; then
      curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz >&2
      tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${TOOLCHAINS_DIR} >&2
    fi
    ARMCC_ROOT=${TOOLCHAINS_DIR}/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
    echo "ARMCC_FLAGS=\"-march=armv6 -mfpu=vfp -mfloat-abi=hard -funsafe-math-optimizations \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include \
      -isystem ${ARMCC_ROOT}/lib/gcc/arm-linux-gnueabihf/8.3.0/include-fixed \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/include/c++/8.3.0 \
      -isystem ${ARMCC_ROOT}/arm-linux-gnueabihf/libc/usr/include \
      -isystem \"\${CROSSTOOL_PYTHON_INCLUDE_PATH}\" \
      -isystem /usr/include\""
    echo "ARMCC_PREFIX=${ARMCC_ROOT}/bin/arm-linux-gnueabihf-"
    ;;
  *)
    echo "Usage: download_toolchains.sh [armhf|aarch64|rpi0]" >&2
    exit
    ;;
  esac

echo "download_toolchains.sh completed successfully." >&2
