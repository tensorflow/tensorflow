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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "abi_version_extension",
    srcs = ["abi_version_extension.cc"],
    hdrs = ["abi_version_extension.h"],
    deps = [
        "//xla/pjrt:pjrt_abi_version",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_abi_version_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/proto:pjrt_abi_version_proto_cc",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "gpu_abi_version_extension",
    srcs = ["gpu_abi_version_extension.cc"],
    hdrs = ["gpu_abi_version_extension.h"],
    deps = [
        ":abi_version_extension",
        "//xla/pjrt/c:pjrt_c_api_abi_version_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/gpu:se_gpu_pjrt_runtime_abi_version",
        "//xla/pjrt/proto:pjrt_abi_version_proto_cc",
        "//xla/pjrt/se:stream_executor_pjrt_abi_version",
        "//xla/stream_executor/abi:runtime_abi_version_manager",
    ],
)

xla_cc_test(
    name = "abi_version_extension_test",
    srcs = ["abi_version_extension_test.cc"],
    deps = [
        ":abi_version_extension",
        "//xla/pjrt:pjrt_abi_version",
        "//xla/pjrt/c:pjrt_c_api_abi_version_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/proto:pjrt_abi_version_proto_cc",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
