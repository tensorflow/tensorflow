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
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_gumgraph_node",
    hdrs = ["hlo_gumgraph_node.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "hlo_gumgraph",
    srcs = ["hlo_gumgraph.cc"],
    hdrs = ["hlo_gumgraph.h"],
    deps = [
        ":hlo_gumgraph_node",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph/analysis:hlo_value_tracing",
        "//xla/hlo/tools/hlo_diff/graph/utils:cycle_detector",
        "//xla/hlo/tools/hlo_diff/utils:hlo_diff_util",
        "//xla/service:call_graph",
        "//xla/service:hlo_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_test",
    srcs = ["hlo_gumgraph_test.cc"],
    deps = [
        ":hlo_gumgraph",
        ":hlo_gumgraph_node",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_value",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
