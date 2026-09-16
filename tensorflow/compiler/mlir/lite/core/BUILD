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
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts_warnings")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

exports_files(
    [
        "model_builder_base.h",
    ],
    visibility = ["//tensorflow/lite/core:__pkg__"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_builder_base",
    srcs = ["model_builder_base.cc"],
    hdrs = ["model_builder_base.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = [
        ":macros",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "//tensorflow/compiler/mlir/lite/core/api:verifier",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "absl_error_model_builder",
    srcs = ["absl_error_model_builder.cc"],
    hdrs = ["absl_error_model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = [
        ":model_builder_base",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "@com_google_absl//absl/log",
    ],
)
