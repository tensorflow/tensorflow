!/bin/bash
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
set -x

if [[ -z "${TF_VERSION}" ]]; then
  echo "TF_VERSION needs to be set, for example 2.1.0rc0."
  exit
fi

VERSIONED_UBUNTU16_CPU_IMAGE="tensorflow/tensorflow:${TF_VERSION}-custom-op-ubuntu16"
VERSIONED_UBUNTU16_GPU_IMAGE="tensorflow/tensorflow:${TF_VERSION}-custom-op-gpu-ubuntu16"

# Build the docker image
cd tensorflow/tools/ci_build
docker build --build-arg TF_PACKAGE_VERSION=${TF_VERSION} --no-cache -t "${VERSIONED_UBUNTU16_CPU_IMAGE}" -f Dockerfile.custom_op_ubuntu_16 .
docker build --build-arg TF_PACKAGE_VERSION=${TF_VERSION} --no-cache -t "${VERSIONED_UBUNTU16_GPU_IMAGE}" -f Dockerfile.custom_op_ubuntu_16_cuda10.1 .

# Log into docker hub, push the image and log out
docker login -u "${TF_DOCKER_USERNAME}" -p "${TF_DOCKER_PASSWORD}"

docker push "${VERSIONED_UBUNTU16_CPU_IMAGE}"
docker push "${VERSIONED_UBUNTU16_GPU_IMAGE}"

# Tag and push the default images for official TF releases
if [[ ${TF_VERSION} == *"rc"* ]]; then
  echo "Do not update default images as ${TF_VERSION} is a release candidate."
else
  UBUNTU16_CPU_IMAGE="tensorflow/tensorflow:custom-op-ubuntu16"
  UBUNTU16_GPU_IMAGE="tensorflow/tensorflow:custom-op-gpu-ubuntu16"

  docker tag "${VERSIONED_UBUNTU16_CPU_IMAGE}" "${UBUNTU16_CPU_IMAGE}"
  docker push "${UBUNTU16_CPU_IMAGE}"

  docker tag "${VERSIONED_UBUNTU16_GPU_IMAGE}" "${UBUNTU16_GPU_IMAGE}"
  docker push "${UBUNTU16_GPU_IMAGE}"
fi

docker logout#!/bin/bash
