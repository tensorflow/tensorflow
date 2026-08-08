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

# Tools to strip out eligible buffers (tensor data) from a TFLite flatbuffer, and load them with
# random values at runtime.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "stripping_lib",
    srcs = ["stripping_lib.cc"],
    hdrs = ["stripping_lib.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_binary(
    name = "strip_buffers_from_fb",
    srcs = ["strip_buffers_from_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_binary(
    name = "reconstitute_buffers_into_fb",
    srcs = ["reconstitute_buffers_into_fb.cc"],
    copts = tflite_copts(),
    linkopts = tflite_linkopts(),
    deps = [
        ":stripping_lib",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
    ],
)
