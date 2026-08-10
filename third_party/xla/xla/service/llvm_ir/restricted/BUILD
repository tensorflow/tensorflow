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

load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

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

xla_cc_test(
    name = "llvm_util_test",
    srcs = ["llvm_util_test.cc"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("llvm_util_test"),
])
