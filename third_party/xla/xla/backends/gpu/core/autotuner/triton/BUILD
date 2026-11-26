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

# Description:
#   Components that help with Triton autotuning.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

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
    name = "dot_search_space",
    srcs = ["dot_search_space.cc"],
    hdrs = ["dot_search_space.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/triton:tma_utils",
        "//xla/backends/gpu/cost_model:gpu_performance_model_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:tma_metadata",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tsl/lib/core:bits",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "dot_search_space_test",
    srcs = ["dot_search_space_test.cc"],
    tags = ["gpu"],
    deps = [
        ":dot_search_space",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "triton_configs",
    srcs = ["triton_configs.cc"],
    hdrs = ["triton_configs.h"],
    deps = [
        "//xla:autotuning_proto_cc",
        "//xla/backends/gpu/autotuner/triton:embed_default_configs",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_cc_test(
    name = "triton_configs_test",
    srcs = ["triton_configs_test.cc"],
    deps = [
        ":triton_configs",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cost_model_config_optimization",
    srcs = ["cost_model_config_optimization.cc"],
    hdrs = ["cost_model_config_optimization.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/cost_model:fusion_analysis_cache",
        "//xla/backends/gpu/cost_model:gpu_indexing_performance_model",
        "//xla/backends/gpu/cost_model:gpu_performance_model_base",
        "//xla/backends/gpu/hlo/transforms:convert_triton_gemm_config",
        "//xla/codegen/xtile:block_level_parameters",
        "//xla/codegen/xtile:xtile_config_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/util:sorted_range",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "cost_model_config_optimization_test",
    srcs = ["cost_model_config_optimization_test.cc"],
    deps = [
        ":cost_model_config_optimization",
        "//xla:autotuning_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//mlir:IR",
    ],
)
