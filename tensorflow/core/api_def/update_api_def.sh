#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# Script to create tensorflow/core/api_def/base_api/api_def*.pbtxt
# files for new ops.

set -e

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

bazel build //tensorflow/core/api_def:update_api_def
bazel-bin/tensorflow/core/api_def/update_api_def \
  --api_def_dir="${current_dir}/base_api" \
  --op_file_pattern="${current_dir}/../ops/*_ops.cc"
