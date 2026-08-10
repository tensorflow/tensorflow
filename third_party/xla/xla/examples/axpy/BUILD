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

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

xla_cc_test(
    name = "stablehlo_compile_test",
    srcs = ["stablehlo_compile_test.cc"],
    data = ["stablehlo_axpy.mlir"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/pjrt:maybe_owning_mlir_module",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_cpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:path",
    ],
)
