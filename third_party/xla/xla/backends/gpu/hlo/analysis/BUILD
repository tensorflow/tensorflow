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
        "//xla:shape_util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_operand_index",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:hlo_fusion_analysis",
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
