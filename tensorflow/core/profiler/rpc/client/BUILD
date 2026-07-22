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

load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "save_profile",
    hdrs = ["save_profile.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/rpc/client:save_profile",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_client_for_pybind",
    actual = ":profiler_client",
)

cc_library(
    name = "profiler_client",
    hdrs = ["profiler_client.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":profiler_client_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
    ],
)

# Linked to pywrap_tensorflow to avoid ODR violation due to tf_grpc_cc_dependencies().
cc_library(
    name = "profiler_client_impl",
    srcs = [
        "profiler_client.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "@tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = True,
)
