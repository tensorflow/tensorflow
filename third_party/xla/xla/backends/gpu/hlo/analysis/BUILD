# Copyright 2026 The OpenXLA Authors.
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
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

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
    name = "alias_info",
    srcs = ["alias_info.cc"],
    hdrs = ["alias_info.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":hlo_fusion_analysis",
        "//xla:shape_util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_operand_index",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "alias_info_test",
    srcs = ["alias_info_test.cc"],
    deps = [
        ":alias_info",
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "nvptx_alias_info",
    srcs = ["nvptx_alias_info.cc"],
    hdrs = ["nvptx_alias_info.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":alias_info",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
    ],
)

xla_cc_test(
    name = "nvptx_alias_info_test",
    srcs = ["nvptx_alias_info_test.cc"],
    deps = [
        ":nvptx_alias_info",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "reduction_utils",
    srcs = ["reduction_utils.cc"],
    hdrs = ["reduction_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "reduction_utils_test",
    srcs = ["reduction_utils_test.cc"],
    deps = [
        ":reduction_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "hlo_fusion_analysis",
    srcs = ["hlo_fusion_analysis.cc"],
    hdrs = ["hlo_fusion_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":reduction_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen:hlo_fusion_spec",
        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:decision",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "hlo_fusion_analysis_test",
    srcs = ["hlo_fusion_analysis_test.cc"],
    deps = [
        ":hlo_fusion_analysis",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_fusible",
    srcs = ["gpu_fusible.cc"],
    hdrs = ["gpu_fusible.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":alias_info",
        ":hlo_fusion_analysis",
        ":reduction_utils",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/hlo/transforms:instruction_fusion",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "gpu_fusible_test",
    srcs = ["gpu_fusible_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":alias_info",
        ":gpu_fusible",
        ":hlo_fusion_analysis",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms:instruction_fusion",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_ordering",
    srcs = ["gpu_hlo_ordering.cc"],
    hdrs = ["gpu_hlo_ordering.h"],
    deps = [
        "//xla:shape_util",
        "//xla/ffi:ffi_registry",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_hlo_ordering_test",
    size = "small",
    srcs = ["gpu_hlo_ordering_test.cc"],
    deps = [
        ":gpu_hlo_ordering",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",  # buildcleaner: keep
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
