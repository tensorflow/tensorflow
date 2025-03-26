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
workspace=$1
cd $workspace/xla
bazel_version=$(head -1 .bazelversion)
bazel_base_url="https://github.com/bazelbuild/bazel/releases/download"
bazel_url=$bazel_base_url/$bazel_version/"bazel-$bazel_version-linux-x86_64"
echo $bazel_url
mkdir -p $workspace/bazel
cd $workspace/bazel
wget $bazel_url
bazel_bin=$(ls)
chmod +x $bazel_bin
