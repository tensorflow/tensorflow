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
# C API for TensorFlow, for use by client language bindings.

load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")

# buildifier: disable=same-origin-load
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# Public targets

filegroup(
    name = "headers",
    srcs = [
        "tsl_status.h",
        "tsl_status_helper.h",
        "tsl_status_internal.h",
    ],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "*test*",
        ],
    ),
    visibility = internal_visibility([
        "//tensorflow/c:__subpackages__",
    ]),
)

cc_library(
    name = "c_api",
    hdrs = [
        "tsl_status.h",
    ],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
    ],
)

cc_library(
    name = "tsl_status_internal",
    hdrs = [
        "tsl_status.h",
        "tsl_status_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "tsl_status",
    srcs = ["tsl_status.cc"],
    hdrs = ["tsl_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tsl_cc_test(
    name = "tsl_status_test",
    srcs = ["tsl_status_test.cc"],
    deps = [
        ":tsl_status",  # buildcleaner: keep
        ":tsl_status_internal",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tsl_status_headers",
    hdrs = ["tsl_status.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tsl_status_helper",
    srcs = ["tsl_status_helper.cc"],
    hdrs = ["tsl_status_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
    ],
)

filegroup(
    name = "tsl_status_internal_headers",
    srcs = ["tsl_status_internal.h"],
    visibility = internal_visibility([
        "//tensorflow/c:__subpackages__",
    ]),
)
