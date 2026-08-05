# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "sm_bw_utils",
    hdrs = ["sm_bw_utils.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:logging",
    ],
)

cuda_library(
    name = "sm_bw_kernels",
    srcs = ["sm_bw_kernels.cu.cc"],
    hdrs = ["sm_bw_kernels.h"],
    tags = ["cuda-only"],
    deps = [
        ":sm_bw_utils",
    ],
)

xla_cc_test(
    name = "sm_bw_test",
    srcs = ["sm_bw_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":sm_bw_kernels",
        ":sm_bw_utils",
        "//xla/tsl/cuda:cudart",
        "@com_google_googletest//:gtest_main",
    ],
)
