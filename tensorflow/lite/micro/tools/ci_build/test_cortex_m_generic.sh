#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#
# Tests the microcontroller code using a Cortex-M4/M4F platform.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=cortex_m_generic
OPTIMIZED_KERNEL_DIR=cmsis_nn

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=cortex-m4 third_party_downloads

# Build for Cortex-M4 (no FPU) without CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=cortex-m4 microlite

# Build for Cortex-M4F (FPU present) without CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=cortex-m4+fp microlite

# Build for Cortex-M4 (no FPU) with CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=cortex-m4 microlite

# Build for Cortex-M4 (FPU present) with CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=cortex-m4+fp microlite
