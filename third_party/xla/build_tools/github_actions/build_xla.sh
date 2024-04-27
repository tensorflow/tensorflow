#!/bin/bash

# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================

echo "XLA run_hlo_module script is running..."

# Build run_hlo_module
build_start_time="$(date +%s)"
echo "run_hlo_module build start time: ${build_start_time}"
bazel build -c opt --keep_going xla/tools:run_hlo_module
build_end_time="$(date +%s)"
echo "run_hlo_module build end time: ${build_end_time}"
build_time="$((build_end_time - build_start_time))"
echo "Build time is ${build_time} seconds."

# Run run_hlo_module
num_iterations=5
run_start_time="$(date +%s)"
echo "run_hlo_module execution start time: ${run_start_time}"
# TODO(b/277240370): use `run_hlo_module`'s timing utils instead of `date`.
bazel run -c opt xla/tools:run_hlo_module -- \
    --input_format=hlo \
    --platform=CPU \
    --iterations=$num_iterations \
    --reference_platform= \
    xla/tools/data/benchmarking/mobilenet_v2.hlo
# add sleep to test base vs PR
sleep 60  # TODO(b/277240370): remove this
run_end_time="$(date +%s)"
echo "run_hlo_module execution end time: ${run_end_time}"
runtime="$((run_end_time - run_start_time))"
echo "Run time for ${num_iterations} iterations is ${runtime} seconds."
