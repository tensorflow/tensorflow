# Description: SPMD partitioning pass.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "spmd_partitioner",
    srcs = [
        "convolution_handler.cc",
        "custom_call_handler.cc",
        "dot_handler.cc",
        "fft_handler.cc",
        "gather_scatter_handler.cc",
        "spmd_partitioner.cc",
        "spmd_partitioner_util.cc",
    ],
    hdrs = [
        "convolution_handler.h",
        "custom_call_handler.h",
        "spmd_partitioner.h",
        "spmd_partitioner_util.h",
    ],
    deps = [
        "//xla:array",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:comparators",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_reachability",
        "//xla/hlo/utils:hlo_query",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_layout",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service:dot_as_convolution_util",
        "//xla/service:flatten_call_graph",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_lexer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:sharding_propagation",
        "//xla/service:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "spmd_partitioner_test",
    srcs = ["spmd_partitioner_test.cc"],
    deps = [
        ":spmd_partitioner",
        ":spmd_prepare",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:sharding_format_picker",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "canonicalize_all_gather_for_cse_test",
    srcs = ["canonicalize_all_gather_for_cse_test.cc"],
    deps = [
        ":canonicalize_all_gather_for_cse",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "shard_barrier_partitioner",
    srcs = ["shard_barrier_partitioner.cc"],
    hdrs = ["shard_barrier_partitioner.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:custom_call_sharding_helper",
    ],
    alwayslink = 1,
)

cc_library(
    name = "canonicalize_all_gather_for_cse",
    srcs = ["canonicalize_all_gather_for_cse.cc"],
    hdrs = ["canonicalize_all_gather_for_cse.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "schedule_aware_collective_ops_cse_test",
    srcs = ["schedule_aware_collective_ops_cse_test.cc"],
    deps = [
        ":schedule_aware_collective_ops_cse",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "schedule_aware_collective_ops_cse",
    srcs = ["schedule_aware_collective_ops_cse.cc"],
    hdrs = ["schedule_aware_collective_ops_cse.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "spmd_prepare_test",
    srcs = ["spmd_prepare_test.cc"],
    deps = [
        ":spmd_prepare",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "spmd_prepare",
    srcs = ["spmd_prepare.cc"],
    hdrs = ["spmd_prepare.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "stateful_rng_spmd_partitioner",
    srcs = ["stateful_rng_spmd_partitioner.cc"],
    hdrs = ["stateful_rng_spmd_partitioner.h"],
    deps = [
        ":spmd_partitioner",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "stateful_rng_spmd_partitioner_test",
    srcs = ["stateful_rng_spmd_partitioner_test.cc"],
    deps = [
        ":stateful_rng_spmd_partitioner",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:rng_expander",
        "//xla/service:sharding_propagation",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "collective_permute_motion",
    srcs = ["collective_permute_motion.cc"],
    hdrs = ["collective_permute_motion.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:while_loop_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "collective_permute_motion_test",
    srcs = ["collective_permute_motion_test.cc"],
    deps = [
        ":collective_permute_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "partition_assignment",
    srcs = [
        "partition_assignment.cc",
    ],
    hdrs = [
        "partition_assignment.h",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "partition_assignment_test",
    srcs = ["partition_assignment_test.cc"],
    deps = [
        ":partition_assignment",
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "whole_graph_manual_pass",
    srcs = [
        "whole_graph_manual_pass.cc",
    ],
    hdrs = [
        "whole_graph_manual_pass.h",
    ],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "whole_graph_manual_pass_test",
    srcs = ["whole_graph_manual_pass_test.cc"],
    deps = [
        ":whole_graph_manual_pass",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

xla_cc_test(
    name = "spmd_partitioner_util_test",
    srcs = ["spmd_partitioner_util_test.cc"],
    deps = [
        ":spmd_partitioner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "@com_google_googletest//:gtest_main",
    ],
)
