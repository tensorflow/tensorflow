#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

# Print build info, including info related to the machine, OS, build tools
# and TensorFlow source code. This can be used by build tools such as Jenkins.
# All info is printed on a single line, in JSON format, to workaround the
# limitation of Jenkins Description Setter Plugin that multi-line regex is
# not supported.
#
# Usage:
#   print_build_info.sh (CONTAINER_TYPE) (COMMAND)
#     e.g.,
#       print_build_info.sh GPU bazel test -c opt --config=cuda //tensorflow/...

# Information about the command
CONTAINER_TYPE=$1
shift 1
COMMAND=("$@")

# Information about machine and OS
OS=$(uname)
KERNEL=$(uname -r)

ARCH=$(uname -p)
PROCESSOR=$(grep "model name" /proc/cpuinfo | head -1 | awk '{print substr($0, index($0, $4))}')
PROCESSOR_COUNT=$(grep "model name" /proc/cpuinfo | wc -l)

MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2, $3}')
SWAP_TOTAL=$(grep SwapTotal /proc/meminfo | awk '{print $2, $3}')

# Information about build tools
if [[ ! -z $(which bazel) ]]; then
  BAZEL_VER=$(bazel version | head -1)
fi

if [[ ! -z $(which javac) ]]; then
  JAVA_VER=$(javac -version 2>&1 | awk '{print $2}')
fi

if [[ ! -z $(which python) ]]; then
  PYTHON_VER=$(python -V 2>&1 | awk '{print $2}')
fi

if [[ ! -z $(which g++) ]]; then
  GPP_VER=$(g++ --version | head -1)
fi

if [[ ! -z $(which swig) ]]; then
  SWIG_VER=$(swig -version > /dev/null | grep -m 1 . | awk '{print $3}')
fi

# Information about TensorFlow source
TF_FETCH_URL=$(git config --get remote.origin.url)
TF_HEAD=$(git rev-parse HEAD)

# NVIDIA & CUDA info
NVIDIA_DRIVER_VER=""
if [[ -f /proc/driver/nvidia/version ]]; then
  NVIDIA_DRIVER_VER=$(head -1 /proc/driver/nvidia/version | awk '{print $(NF-6)}')
fi

CUDA_DEVICE_COUNT="0"
CUDA_DEVICE_NAMES=""
if [[ ! -z $(which nvidia-debugdump) ]]; then
  CUDA_DEVICE_COUNT=$(nvidia-debugdump -l | grep "^Found [0-9]*.*device.*" | awk '{print $2}')
  CUDA_DEVICE_NAMES=$(nvidia-debugdump -l | grep "Device name:.*" | awk '{print substr($0, index($0,\
 $3)) ","}')
fi

CUDA_TOOLKIT_VER=""
if [[ ! -z $(which nvcc) ]]; then
  CUDA_TOOLKIT_VER=$(nvcc -V | grep release | awk '{print $(NF)}')
fi

# Print info
echo "TF_BUILD_INFO = {"\
"container_type: \"${CONTAINER_TYPE}\", "\
"command: \"${COMMAND[*]}\", "\
"source_HEAD: \"${TF_HEAD}\", "\
"source_remote_origin: \"${TF_FETCH_URL}\", "\
"OS: \"${OS}\", "\
"kernel: \"${KERNEL}\", "\
"architecture: \"${ARCH}\", "\
"processor: \"${PROCESSOR}\", "\
"processor_count: \"${PROCESSOR_COUNT}\", "\
"memory_total: \"${MEM_TOTAL}\", "\
"swap_total: \"${SWAP_TOTAL}\", "\
"Bazel_version: \"${BAZEL_VER}\", "\
"Java_version: \"${JAVA_VER}\", "\
"Python_version: \"${PYTHON_VER}\", "\
"gpp_version: \"${GPP_VER}\", "\
"swig_version: \"${SWIG_VER}\", "\
"NVIDIA_driver_version: \"${NVIDIA_DRIVER_VER}\", "\
"CUDA_device_count: \"${CUDA_DEVICE_COUNT}\", "\
"CUDA_device_names: \"${CUDA_DEVICE_NAMES}\", "\
"CUDA_toolkit_version: \"${CUDA_TOOLKIT_VER}\""\
"}"
