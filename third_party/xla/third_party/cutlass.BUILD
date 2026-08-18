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

# Description:
# CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance
# matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "cutlass_header_files",
    srcs = glob([
        "include/**",
    ]),
)

filegroup(
    name = "cutlass_util_header_files",
    srcs = glob([
        "tools/util/include/**",
    ]),
)

cc_library(
    name = "cutlass",
    hdrs = [
        ":cutlass_header_files",
        ":cutlass_util_header_files",
    ],
    includes = [
        "include",
        "tools/util/include",
    ],
)
