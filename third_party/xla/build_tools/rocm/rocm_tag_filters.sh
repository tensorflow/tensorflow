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
#
# ==============================================================================

TAG_FILTERS=(
    -no_gpu
    -requires-gpu-intel
    -requires-gpu-nvidia
    -cuda-only
    -oneapi-only
    -requires-gpu-sm60
    -requires-gpu-sm60-only
    -requires-gpu-sm70
    -requires-gpu-sm70-only
    -requires-gpu-sm80
    -requires-gpu-sm80-only
    -requires-gpu-sm86
    -requires-gpu-sm86-only
    -requires-gpu-sm89
    -requires-gpu-sm89-only
    -requires-gpu-sm90
    -requires-gpu-sm90-only
    -skip_rocprofiler_sdk
    -no_oss
    -oss_excluded
    -oss_serial
)

echo $(IFS=, ; echo "${TAG_FILTERS[*]}")
