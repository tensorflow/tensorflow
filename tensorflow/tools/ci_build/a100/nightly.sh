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


docker pull tensorflow/tensorflow:devel-gpu
docker run --gpus all -w /tensorflow_src -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:devel-gpu bash -c "git pull; bazel test --config=cuda -c opt --test_tag_filters=gpu,-no_gpu,-benchmark-test,-no_oss,-oss_excluded,-oss_serial,-v1only,-no_gpu_presubmit,-no_cuda11 -- //tensorflow/... -//tensorflow/compiler/tf2tensorrt/... -//tensorflow/compiler/mlir/tosa/... //tensorflow/compiler/mlir/lite/... -//tensorflow/lite/micro/examples/... -//tensorflow/core/tpu/... -//tensorflow/lite/..."
