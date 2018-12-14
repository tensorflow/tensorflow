#!/bin/bash
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

TARGET="$1"
OUTPUT="$2"

if [[ -z "${TARGET}" || -z "${OUTPUT}" ]]; then
  echo "Usage:"
  echo "$0 <target> <output>"
  exit 1
fi

TEMPDIR="$(mktemp -d)"
ROOT="${PWD}"
PKG="third_party/toolchains/preconfig"
IFS='-' read -ra PLATFORM <<< "${TARGET}"
OS="${PLATFORM[0]}"
PY_VERSION="${PLATFORM[1]}"
COMPILER="${PLATFORM[2]}"
CUDA_VERSION="${PLATFORM[3]}"
CUDNN_VERSION="${PLATFORM[4]}"
TENSORRT_VERSION="${PLATFORM[5]}"

# TODO(klimek): Put this into the name.

if [[ "${COMPILER}" == "gcc" ]]; then
  COMPILER="gcc-nvcc-${CUDA_VERSION}"
fi

echo "OS: ${OS}"
echo "Python: ${PY_VERSION}"
echo "Compiler: ${COMPILER}"
echo "CUDA: ${CUDA_VERSION}"
echo "CUDNN: ${CUDNN_VERSION}"
echo "TensorRT: ${TENSORRT_VERSION}"

bazel build --define=mount_project="${PWD}" "${PKG}/generate:${TARGET}"
cd "${TEMPDIR}"
tar xvf "${ROOT}/bazel-bin/${PKG}/generate/${TARGET}_outputs.tar"

# Delete all empty files: configurations leave empty files around when they are
# unnecessary.
find . -empty -delete

# We build up the following directory structure with preconfigured packages:
# <OS>/
#   <CUDA>-<CUDNN>/
#   <COMPILER>/
#   <PYTHON>/
#   <TENSORRT>/

# Create our toplevel output directory for the OS.
mkdir "${OS}"

# Python:
mv local_config_python "${OS}/${PY_VERSION}"

# Compiler:
mv local_config_cuda/crosstool "${OS}/${COMPILER}"

# CUDA:
mv local_config_cuda "${OS}/${CUDA_VERSION}-${CUDNN_VERSION}"

# TensorRT:
mv local_config_tensorrt "${OS}/${TENSORRT_VERSION}"

# Cleanup for copybara.
find "${OS}" -name 'BUILD' -o -name '*.bzl' |xargs buildifier
find "${OS}" -name 'BUILD' -o -name '*.bzl' |xargs -I {} mv {} {}.oss

# Tar it up:
tar cvf "${OUTPUT}" "${OS}"

