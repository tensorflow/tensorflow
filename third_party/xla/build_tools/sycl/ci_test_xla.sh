#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
#set the configuration to select the host compiler
./configure.py --backend=SYCL --host_compiler=GCC
bazel build \
      --config=sycl_hermetic \
      --build_tag_filters=gpu,sycl,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-sycl \
      --test_tag_filters=gpu,sycl,requires-gpu-intel,-requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-cuda-only,-rocm-only,-no-sycl \
      //xla/stream_executor/sycl:stream_executor_sycl
