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

if [ -z ${SYCL_TOOLKIT_PATH+x} ];
then
  export SYCL_TOOLKIT_PATH=$workspace/oneapi/compiler/2024.1/
fi
bazel_bin=$(ls $workspace/bazel/)
./configure.py --backend=SYCL --host_compiler=GCC
$workspace/bazel/$bazel_bin build --config=verbose_logs -s --verbose_failures --nocheck_visibility //xla/tools:run_hlo_module
