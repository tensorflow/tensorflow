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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

build_test(
    name = "mlir-interpreter-runner_build_test",
    targets = [
        ":mlir-interpreter-runner",
    ],
)

xla_cc_binary(
    name = "mlir-interpreter-runner",
    testonly = True,
    srcs = ["mlir_interpreter_runner.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/dialects",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ParseUtilities",
        "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        "@llvm-project//mlir:RegisterAllPasses",  # buildcleaner: keep
        "@llvm-project//mlir:Support",
    ],
)
