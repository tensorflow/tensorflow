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
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "profiler.h",
        "telemetry_setting.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "profiler",
    hdrs = ["profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [":telemetry_setting"],
)

tflite_cc_library_with_c_headers_test(
    name = "telemetry_setting",
    hdrs = ["telemetry_setting.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_setting_internal",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "telemetry_setting_internal",
    srcs = ["telemetry_setting_internal.cc"],
    hdrs = ["telemetry_setting_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/core/c:common"],
)
