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
# ==============================================================================
#
# Script to produce a tarball release of the C-library, Java native library
# and Java .jars.
# Builds a docker container and then builds in said container.
#
# See libtensorflow_cpu.sh and libtensorflow_gpu.sh

set -ex

# Current script directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/../builds/builds_common.sh"
DOCKER_CONTEXT_PATH="$(realpath ${SCRIPT_DIR}/..)"
ROOT_DIR="$(realpath ${SCRIPT_DIR}/../../../../)"

DOCKER_IMAGE="tf-libtensorflow-cpu"
DOCKER_FILE="Dockerfile.cpu"
DOCKER_BINARY="docker"
if [ "${TF_NEED_CUDA}" == "1" ]; then
  DOCKER_IMAGE="tf-tensorflow-gpu"
  DOCKER_BINARY="nvidia-docker"
  DOCKER_FILE="Dockerfile.gpu"
fi

docker build \
  -t "${DOCKER_IMAGE}" \
  -f "${DOCKER_CONTEXT_PATH}/${DOCKER_FILE}" \
  "${DOCKER_CONTEXT_PATH}"

${DOCKER_BINARY} run \
  --rm \
  --pid=host \
  -v ${ROOT_DIR}:/workspace \
  -w /workspace \
  -e "PYTHON_BIN_PATH=/usr/bin/python" \
  -e "TF_NEED_HDFS=0" \
  -e "TF_NEED_CUDA=${TF_NEED_CUDA}" \
  -e "TF_NEED_OPENCL_SYCL=0" \
  "${DOCKER_IMAGE}" \
  "/workspace/tensorflow/tools/ci_build/linux/libtensorflow.sh"
