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
# Profiler C API

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "headers",
    srcs = [
        "pluggable_profiler.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "pluggable_profiler_hdrs",
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "pluggable_profiler",
    srcs = ["pluggable_profiler.cc"],
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":pluggable_profiler_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/device:device_utils",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "pluggable_profiler_internal",
    hdrs = [
        "pluggable_profiler.h",
        "pluggable_profiler_internal.h",
    ],
    visibility = ["//tensorflow/core/common_runtime/pluggable_device:__subpackages__"],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
