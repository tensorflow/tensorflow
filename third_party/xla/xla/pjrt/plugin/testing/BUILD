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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "testing_cpp_pjrt",
    srcs = [
        "testing_cpp_pjrt.cc",
    ],
    hdrs = ["testing_cpp_pjrt.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "testing_cpp_pjrt_test",
    srcs = ["testing_cpp_pjrt_test.cc"],
    deps = [
        ":testing_cpp_pjrt",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "testing_c_pjrt_internal",
    srcs = [
        "testing_c_pjrt_internal.cc",
    ],
    hdrs = ["testing_c_pjrt_internal.h"],
    deps = [
        ":testing_cpp_pjrt",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "testing_c_pjrt_internal_test",
    srcs = ["testing_c_pjrt_internal_test.cc"],
    deps = [
        ":testing_c_pjrt_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
