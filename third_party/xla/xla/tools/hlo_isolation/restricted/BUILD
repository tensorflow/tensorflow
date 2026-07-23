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

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

sh_test(
    name = "hlo_isolation_cli_test",
    srcs = ["hlo_isolation_cli_test.sh"],
    args = [
        "$(location :hlo_isolation_test)",
        "$(location @llvm-project//llvm:FileCheck)",
        "$(location :hlo_isolation_cli_test_data)",
    ],
    data = [
        ":hlo_isolation_cli_test_data",
        ":hlo_isolation_test",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "hlo_isolation_test",
    testonly = True,
    srcs = ["hlo_isolation_cli.cc"],
    deps = [
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_runner_interface",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:platform",
        "//xla/tools/hlo_isolation:hlo_isolation_api",
        "//xla/tools/hlo_isolation:hlo_isolation_proto_cc",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

filegroup(
    name = "hlo_isolation_cli_test_data",
    srcs = ["//xla/tools:data/hlo_isolation_cli_test.hlo"],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("hlo_isolation_cli_test"),
    xla_restricted_allow("hlo_isolation_cli_test_data"),
    xla_restricted_allow("hlo_isolation_test"),
])
