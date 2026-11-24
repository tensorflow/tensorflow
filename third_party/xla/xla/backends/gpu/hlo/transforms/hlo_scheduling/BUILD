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
load("//xla/tests:build_defs.bzl", "GPU_DEFAULT_BACKENDS", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

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
    name = "gpu_hlo_schedule",
    srcs = ["gpu_hlo_schedule.cc"],
    hdrs = ["gpu_hlo_schedule.h"],
    deps = [
        ":gpu_latency_hiding_scheduler",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/cost_model:analytical_latency_estimator",
        "//xla/backends/gpu/cost_model:gpu_hlo_cost_analysis",
        "//xla/backends/gpu/cost_model:hlo_op_profiles",
        "//xla/backends/gpu/cost_model:sol_latency_estimator",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/backends/gpu/hlo/transforms:pgle_accuracy_checker",
        "//xla/backends/gpu/hlo/transforms:scheduling_instruction_annotator",
        "//xla/backends/gpu/hlo/transforms/collectives:async_collective_annotator",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_ops_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:legalize_scheduling_annotations",
        "//xla/hlo/transforms:scheduler_memory_fencing",
        "//xla/hlo/transforms/collectives:async_collective_creator",
        "//xla/hlo/transforms/collectives:p2p_schedule_preparation",
        "//xla/hlo/transforms/hlo_scheduling:latency_hiding_scheduler",
        "//xla/hlo/transforms/hlo_scheduling:profile_guided_latency_estimator",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:buffer_value",
        "//xla/service:device_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_profiling_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:flag_utils",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

xla_test(
    name = "gpu_hlo_schedule_test",
    srcs = ["gpu_hlo_schedule_test.cc"],
    backends = GPU_DEFAULT_BACKENDS,
    use_legacy_runtime = True,
    deps = [
        ":gpu_hlo_schedule",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:backend",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_profiling_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/stream_executor:device_description",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

cc_library(
    name = "gpu_latency_hiding_scheduler",
    srcs = ["gpu_latency_hiding_scheduler.cc"],
    hdrs = ["gpu_latency_hiding_scheduler.h"],
    deps = [
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/transforms:dynamic_slice_copy",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_ops_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/hlo_scheduling:latency_hiding_scheduler",
        "//xla/hlo/transforms/hlo_scheduling:profile_guided_latency_estimator",
        "//xla/hlo/utils:hlo_query",
        "//xla/runtime:device_id",
        "//xla/service:collective_ops_utils",
        "//xla/service:device_assignment",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gpu_latency_hiding_scheduler_test",
    srcs = ["gpu_latency_hiding_scheduler_test.cc"],
    deps = [
        ":gpu_hlo_schedule",
        ":gpu_latency_hiding_scheduler",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/hlo/analysis:alias_info",
        "//xla/backends/gpu/hlo/transforms/collectives:collective_domain",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/hlo_scheduling:latency_hiding_scheduler",
        "//xla/hlo/transforms/hlo_scheduling:profile_guided_latency_estimator",
        "//xla/service:device_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",  # fixdeps: keep
        "@llvm-project//mlir:IR",
    ],
)
