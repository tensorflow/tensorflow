#!/bin/bash
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
set -e

# Build the docker image
cd tensorflow/tools/ci_build
docker build -t gpu_test_container:latest -f \
   Dockerfile.rbe.cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython .

DEFAULT_BAZEL_TARGETS="//tensorflow/... -//tensorflow/python/integration_testing/... -//tensorflow/compiler/tf2tensorrt/... -//tensorflow/compiler/mlir/tosa/... -//tensorflow/compiler/xrt/... //tensorflow/compiler/mlir/lite/... -//tensorflow/lite/micro/examples/... -//tensorflow/core/tpu/..."

docker run --rm \
  --gpus all \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  gpu_test_container:latest bash -c "bazel test --config=rbe_linux_cuda11.0_nvcc_py3.6 --config=tensorflow_testing_rbe_linux --test_tag_filters=gpu,-no_gpu,-nogpu,-benchmark-test,-no_oss,-oss_serial,-v1only,-no_gpu_presubmit,-no_cuda11 -- ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/..."

