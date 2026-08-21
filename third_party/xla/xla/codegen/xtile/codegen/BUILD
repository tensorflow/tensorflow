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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

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
    name = "emitter_helpers",
    srcs = ["emitter_helpers.cc"],
    hdrs = ["emitter_helpers.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/codegen:ir_emission_utils",
        "//xla/codegen/emitters:elemental_hlo_to_mlir",
        "//xla/codegen/tiling/experimental:reshape_analysis",
        "//xla/codegen/tiling/experimental:scheduling",
        "//xla/codegen/tiling/experimental:tile",
        "//xla/codegen/tiling/experimental:tiled_hlo",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:transformation_helpers",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "fusion_emitter",
    srcs = [
        "experimental_fusion_emitter.cc",
    ],
    hdrs = [
        "experimental_fusion_emitter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":dot_algorithms",
        ":emitter_helpers",
        "//xla:autotuning_proto_cc",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/tiling/experimental:reshape_analysis",
        "//xla/codegen/tiling/experimental:scheduling",
        "//xla/codegen/tiling/experimental:tile",
        "//xla/codegen/tiling/experimental:tiled_hlo",
        "//xla/codegen/xtile:xtile_config_proto_cc",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/codegen/xtile/ir/transforms:passes",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "dot_algorithms",
    srcs = ["dot_algorithms.cc"],
    hdrs = ["dot_algorithms.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":emitter_helpers",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "//xla/service:algorithm_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

xla_cc_test(
    name = "dot_algorithms_test",
    srcs = ["dot_algorithms_test.cc"],
    deps = [
        ":dot_algorithms",
        ":emitter_helpers",
        "//xla:xla_proto_cc",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
