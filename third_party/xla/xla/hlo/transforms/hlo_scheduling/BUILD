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
#   Implementation of XLA's Latency Hiding Scheduler (LHS) and supporting
#   components.

load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "approximate_latency_estimator",
    srcs = ["approximate_latency_estimator.cc"],
    hdrs = ["approximate_latency_estimator.h"],
    deps = [
        ":hlo_schedule_graph",
        ":latency_estimator",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "async_tracker",
    srcs = ["async_tracker.cc"],
    hdrs = ["async_tracker.h"],
    deps = [
        ":latency_estimator",
        ":scheduler_config",
        ":scheduler_core",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scheduler_config",
    hdrs = ["scheduler_config.h"],
)

cc_library(
    name = "scheduler_core",
    hdrs = ["scheduler_core.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "hlo_schedule_graph",
    srcs = ["hlo_schedule_graph.cc"],
    hdrs = ["hlo_schedule_graph.h"],
    deps = [
        ":async_tracker",
        ":latency_estimator",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "latency_estimator",
    srcs = ["latency_estimator.cc"],
    hdrs = ["latency_estimator.h"],
    deps = [
        "//xla:frontend_attributes",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "latency_hiding_scheduler",
    srcs = ["latency_hiding_scheduler.cc"],
    hdrs = ["latency_hiding_scheduler.h"],
    deps = [
        ":approximate_latency_estimator",
        ":async_tracker",
        ":hlo_schedule_graph",
        ":latency_estimator",
        ":scheduler_config",
        ":scheduler_core",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:buffer_value",
        "//xla/service:collective_ops_utils",
        "//xla/service:dump",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/service:scheduling_annotations_util",
        "//xla/service/heap_simulator",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
    ],
)

xla_cc_test(
    name = "latency_hiding_scheduler_test",
    srcs = ["latency_hiding_scheduler_test.cc"],
    deps = [
        ":latency_hiding_scheduler",
        "//xla:debug_options_flags",
        "//xla:frontend_attributes",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms:legalize_scheduling_annotations",
        "//xla/hlo/transforms/collectives:async_collective_creator",
        "//xla/service:device_assignment",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profile_guided_latency_estimator",
    srcs = ["profile_guided_latency_estimator.cc"],
    hdrs = ["profile_guided_latency_estimator.h"],
    deps = [
        ":latency_hiding_scheduler",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
    ],
)

xla_cc_test(
    name = "profile_guided_latency_estimator_test",
    srcs = ["profile_guided_latency_estimator_test.cc"],
    deps = [
        ":latency_hiding_scheduler",
        ":profile_guided_latency_estimator",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)
