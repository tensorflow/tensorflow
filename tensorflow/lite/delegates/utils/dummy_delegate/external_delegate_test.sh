#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set -o errexit
set -o nounset

readonly benchmark_tool=tensorflow/lite/tools/benchmark/benchmark_model
readonly external_delegate=tensorflow/lite/delegates/utils/dummy_delegate/dummy_external_delegate.so
readonly model=external/tflite_mobilenet_float/mobilenet_v1_1.0_224.tflite
readonly benchmark_log=/tmp/benchmark.out

die() { echo "$@" >&2; exit 1; }

$benchmark_tool --graph=$model \
    --external_delegate_path=$external_delegate \
    --external_delegate_options='error_during_init:true;error_during_prepare:true' \
    >& $benchmark_log
cat $benchmark_log
grep -q 'EXTERNAL delegate created.' $benchmark_log \
    || die "Didn't find expected log contents"

echo "PASS"
