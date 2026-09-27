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

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

BASE_COPTS = [
    "-fexceptions",
    "-Wno-unused-variable",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-reorder-ctor",
    "-Wno-non-virtual-dtor",
    "-Wno-uninitialized",
    "-Wno-pass-failed",
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    "-DRAFT_SYSTEM_LITTLE_ENDIAN",
]

cuda_library(
    name = "raft_matrix",
    copts = BASE_COPTS,
    includes = [
        "cpp/include",
        "cpp/internal",
    ],
    textual_hdrs = glob([
        "cpp/include/**/*.cuh",
        "cpp/include/**/*.hpp",
        "cpp/internal/**/*.cuh",
    ]) + [
        "cpp/include/raft/compat/clang_cuda_intrinsics.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@rapids_logger",
        "@rmm",
    ],
)

cuda_library(
    name = "select_k_runner",
    srcs = ["select_k_runner.cu.cc"],
    hdrs = ["select_k_runner.hpp"],
    copts = BASE_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":raft_matrix",
    ],
)

cc_test(
    name = "select_k_smoke_test",
    srcs = ["select_k_smoke_test.cu.cc"],
    copts = BASE_COPTS,
    deps = [
        ":select_k_runner",
        "@com_google_googletest//:gtest_main",
    ],
)
