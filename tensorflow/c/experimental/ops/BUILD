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

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental ops. These will eventually be replaced by machine-generated versions.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "array_ops",
    srcs = [
        "array_ops.cc",
    ],
    hdrs = [
        "array_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "io_ops",
    srcs = [
        "io_ops.cc",
    ],
    hdrs = [
        "io_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "math_ops",
    srcs = [
        "math_ops.cc",
    ],
    hdrs = [
        "math_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "nn_ops",
    srcs = [
        "nn_ops.cc",
    ],
    hdrs = [
        "nn_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "resource_variable_ops",
    srcs = [
        "resource_variable_ops.cc",
    ],
    hdrs = [
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ops",
    hdrs = [
        "array_ops.h",
        "io_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_ops",
        ":io_ops",
        ":math_ops",
        ":nn_ops",
        ":resource_variable_ops",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_ops.h",
        "io_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = ["//tensorflow/python:__subpackages__"],
)
