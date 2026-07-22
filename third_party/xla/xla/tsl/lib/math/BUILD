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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow:__subpackages__",
    ]),
    licenses = ["notice"],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "math_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tsl_cc_test(
    name = "math_util_test",
    size = "small",
    srcs = [
        "math_util_test.cc",
    ],
    deps = [
        ":math_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:types",
        "@com_google_googletest//:gtest_main",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "math_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

exports_files([
    "math_util.h",
])
