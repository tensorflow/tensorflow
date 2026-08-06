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
load("//xla/tsl:tsl.bzl", "if_windows")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

cc_library(
    name = "dialects",
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "util.cc",
            "symbol_finder_linux.cc",
            "symbol_finder_windows.cc",
            "symbol_finder_test.cc",
        ],
    ),
    deps = [
        ":dialect_utils",
        ":symbol_finder",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dialect_utils",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "comparators.h",
        "cwise_math.h",
        "util.h",
    ],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "symbol_finder_linux",
    srcs = [
        "symbol_finder.h",
        "symbol_finder_linux.cc",
    ],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "symbol_finder_windows",
    srcs = [
        "symbol_finder.h",
        "symbol_finder_windows.cc",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "symbol_finder",
    hdrs = ["symbol_finder.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ] + if_windows(
        [":symbol_finder_windows"],
        [":symbol_finder_linux"],
    ),
)

xla_cc_test(
    name = "symbol_finder_test",
    srcs = ["symbol_finder_test.cc"],
    deps = [
        ":symbol_finder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
