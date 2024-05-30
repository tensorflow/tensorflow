# Description:
#   XLA Heap simulator.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)

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
    name = "allocation_block",
    srcs = ["allocation_block.cc"],
    hdrs = ["allocation_block.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "heap_simulator",
    srcs = ["heap_simulator.cc"],
    hdrs = ["heap_simulator.h"],
    deps = [
        ":allocation_block",
        "//xla:comparison_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:buffer_value",
        "//xla/service:buffer_value_containers",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service:time_utils",
        "//xla/service:tuple_points_to_analysis",
        "//xla/service/memory_space_assignment:repacking",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "heap_simulator_test",
    srcs = ["heap_simulator_test.cc"],
    deps = [
        ":allocation_block",
        ":heap_simulator",
        "//xla:literal",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_value",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_value",
        "//xla/service:tuple_points_to_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
)
