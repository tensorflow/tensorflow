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
load("//xla/tests:build_defs.bzl", "xla_test")
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

xla_test(
    name = "cpu_aot_compiler_test",
    srcs = ["cpu_aot_compiler_test.cc"],
    backends = [
        "cpu",
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_runner_interface",
        "//xla/service/cpu:cpu_aot_compilation_result",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@tsl//tsl/platform:casts",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("cpu_aot_compiler_test"),
])
