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
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

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

tf_proto_library(
    name = "xtile_config_proto",
    srcs = ["xtile_config.proto"],
    make_default_target_header_only = True,
)

cc_library(
    name = "tiling_from_block_parameters",
    srcs = ["tiling_from_block_parameters.cc"],
    hdrs = [
        "tiling_from_block_parameters.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":block_level_parameters",
        ":xtile_config_proto_cc",
        "//xla:status_macros",
        "//xla:util",
        "//xla/codegen/tiling:symbolic_tile_analysis",
        "//xla/codegen/tiling:tiling_specification",
        "//xla/codegen/tiling/experimental:tile",
        "//xla/codegen/tiling/experimental:tiling_space_utils",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "tiling_from_block_parameters_test",
    srcs = ["tiling_from_block_parameters_test.cc"],
    deps = [
        ":block_level_parameters",
        ":tiling_from_block_parameters",
        ":xtile_config_proto_cc",
        "//xla:status_macros",
        "//xla/codegen/tiling:symbolic_tile_analysis",
        "//xla/codegen/tiling:tiling_specification",
        "//xla/codegen/tiling/experimental:tile",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "block_level_parameters",
    hdrs = [
        "block_level_parameters.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xtile_config_proto_cc",
        "//xla/service/gpu:backend_configs_cc",
    ],
)

xla_cc_test(
    name = "block_level_parameters_test",
    srcs = ["block_level_parameters_test.cc"],
    deps = [
        ":block_level_parameters",
        ":xtile_config_proto_cc",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
