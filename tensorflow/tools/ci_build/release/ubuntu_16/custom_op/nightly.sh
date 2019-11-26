#!/bin/bash
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
set -e

# 1. Build the test server
UBUNTU16_CPU_IMAGE="tensorflow/tensorflow:nightly-custom-op-ubuntu16"
UBUNTU16_GPU_IMAGE="tensorflow/tensorflow:nightly-custom-op-gpu-ubuntu16"

# Build the docker image
cd tensorflow/tools/ci_build
docker build --no-cache -t "${UBUNTU16_CPU_IMAGE}" -f Dockerfile.custom_op_ubuntu_16 .
docker build --no-cache -t "${UBUNTU16_GPU_IMAGE}" -f Dockerfile.custom_op_ubuntu_16_gpu .

# Log into docker hub, push the image and log out
docker login -u "${TF_DOCKER_USERNAME}" -p "${TF_DOCKER_PASSWORD}"

docker push "${UBUNTU16_CPU_IMAGE}"
docker push "${UBUNTU16_GPU_IMAGE}"

docker logout#!/bin/bash
