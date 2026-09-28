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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility_macros",
    hdrs = ["compatibility_macros.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "runtime_shape",
    srcs = ["runtime_shape.cc"],
    hdrs = ["runtime_shape.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [":compatibility_macros"],
)

tf_cc_test(
    name = "runtime_shape_test",
    srcs = ["runtime_shape_test.cc"],
    deps = [
        ":runtime_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# LINT.IfChange

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":compatibility_macros",
        ":cppmath",
    ],
)

config_setting(
    name = "x86_32",
    constraint_values = ["@platforms//cpu:x86_32"],
)

config_setting(
    name = "x86_64",
    constraint_values = ["@platforms//cpu:x86_64"],
)

selects.config_setting_group(
    name = "x86_any",
    match_any = [
        ":x86_32",
        ":x86_64",
    ],
)

cc_library(
    name = "cpu_check",
    hdrs = [
        "optimized/neon_check.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        ":x86_any": ["@arm_neon_2_x86_sse"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility_macros",
        ":cpu_check",
        "//tensorflow/compiler/mlir/lite/core:macros",
        "@gemmlowp//:fixedpoint",
    ],
)

# LINT.ThenChange(//tensorflow/lite/kernels/internal/BUILD)
