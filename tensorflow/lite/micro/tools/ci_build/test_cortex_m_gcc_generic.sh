#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

TARGET=cortex_m_gcc_generic

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn TARGET=${TARGET} CORTEX_M_CORE=M4F third_party_downloads

# Build for Cortex-M4 (no FPU) without CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} CORTEX_M_CORE=M4 microlite

# Build for Cortex-M4F (FPU present) without CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} CORTEX_M_CORE=M4F microlite

# Build for Cortex-M4 (no FPU) with CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn TARGET=${TARGET} CORTEX_M_CORE=M4 microlite

# Build for Cortex-M4 (FPU present) with CMSIS
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn TARGET=${TARGET} CORTEX_M_CORE=M4F microlite
