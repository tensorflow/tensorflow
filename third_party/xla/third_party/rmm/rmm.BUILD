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
    "-Wno-unused-result",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-self-move",
    "-Wno-pragma-once-outside-header",
    "-fexceptions",
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
]

cuda_library(
    name = "rmm",
    srcs = glob([
        "cpp/src/*.cpp",
    ]),
    copts = BASE_COPTS,
    features = ["-use_header_modules"],
    includes = ["cpp/include"],
    textual_hdrs = glob([
        "cpp/include/rmm/**/*.h",
        "cpp/include/rmm/**/*.hpp",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@rapids_logger",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cc_test(
    name = "cuda_stream_tests",
    srcs = ["cpp/tests/cuda_stream_tests.cpp"],
    copts = BASE_COPTS,
    deps = [
        ":rmm",
        "@com_google_googletest//:gtest_main",
    ],
)
