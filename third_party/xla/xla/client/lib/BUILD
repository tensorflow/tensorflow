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

# Common computation builders for XLA.

load("//xla/tests:build_defs.bzl", "generate_backend_suites")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/client:friends"]),
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service",
        "//xla/tsl/lib/gtl:value_or_die",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
    ],
)
