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
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/compiler:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    deps = [
        ":ordered_set",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ordered_set",
    hdrs = ["ordered_set.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "graphcycles_test",
    srcs = ["graphcycles_test.cc"],
    deps = [
        ":graphcycles",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "ordered_set_test",
    srcs = ["ordered_set_test.cc"],
    deps = [
        ":ordered_set",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)
