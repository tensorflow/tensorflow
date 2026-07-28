# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

#Description : NVIDIA Tools Extension (NVTX) library for adding profiling annotations to applications.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # NVIDIA proprietary license

filegroup(
    name = "nvtx_header_files",
    srcs = glob([
        "nvtx3/**",
    ]),
)

cc_library(
    name = "nvtx",
    hdrs = [":nvtx_header_files"],
    include_prefix = "third_party",
)
