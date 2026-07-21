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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

tf_proto_library(
    name = "custom_kernel_proto",
    srcs = ["custom_kernel.proto"],
    visibility = [":friends"],
    deps = [
        "//xla/stream_executor:kernel_spec_proto",
        "//xla/stream_executor:launch_dim_proto",
    ],
)

cc_library(
    name = "custom_kernel",
    srcs = ["custom_kernel.cc"],
    hdrs = ["custom_kernel.h"],
    visibility = [":friends"],
    deps = [
        ":custom_kernel_proto_cc",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "custom_kernel_test",
    srcs = ["custom_kernel_test.cc"],
    deps = [
        ":custom_kernel",
        ":custom_kernel_proto_cc",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# PTX Custom Kernels
#===--------------------------------------------------------------------------------------------===#

cc_library(
    name = "ptx_custom_kernel",
    srcs = ["ptx_custom_kernel.cc"],
    hdrs = ["ptx_custom_kernel.h"],
    visibility = [":friends"],
    deps = [
        ":custom_kernel",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_args_packing_spec",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "ptx_custom_kernel_test",
    srcs = ["ptx_custom_kernel_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":custom_kernel",
        ":ptx_custom_kernel",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)
