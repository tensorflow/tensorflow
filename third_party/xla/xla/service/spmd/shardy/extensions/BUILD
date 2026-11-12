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
    name = "partial_reduce_custom_call",
    srcs = ["partial_reduce_custom_call.cc"],
    hdrs = ["partial_reduce_custom_call.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/hlo/builder/lib:approx_topk_shape",
        "//xla/mlir_hlo:unregistered_attributes",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/export:utils",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "custom_call_extensions",
    srcs = ["custom_call_extensions.cc"],
    hdrs = ["custom_call_extensions.h"],
    deps = [
        ":partial_reduce_custom_call",
        "//xla:sharding_op_util",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/mlir_hlo:unregistered_attributes",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "mhlo_extensions",
    srcs = ["mhlo_extensions.cc"],
    hdrs = ["mhlo_extensions.h"],
    deps = [
        "//xla/mlir_hlo",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/export:utils",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)
