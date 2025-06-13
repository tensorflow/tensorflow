# Description: SPMD partitioning pass.

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
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/parser:hlo_lexer",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/utils:hlo_query",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_layout",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service:dot_as_convolution_util",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_module_config",
        "//xla/service:memory_annotations_hdr",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:sharding_propagation",
        "//xla/service/spmd/shardy:constants",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
        "@local_tsl//tsl/platform:numbers",
    ],
)

xla_cc_test(
    name = "spmd_partitioner_test",
    srcs = ["spmd_partitioner_test.cc"],
    shard_count = 10,
    deps = [
        ":sharding_format_picker",
        ":spmd_partitioner",
        ":spmd_prepare",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_verifier",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "sharding_format_picker",
    testonly = True,
    srcs = ["sharding_format_picker.cc"],
    hdrs = ["sharding_format_picker.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "canonicalize_all_gather_for_cse_test",
    srcs = ["canonicalize_all_gather_for_cse_test.cc"],
    deps = [
        ":canonicalize_all_gather_for_cse",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "shard_barrier_partitioner",
    hdrs = ["shard_barrier_partitioner.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:custom_call_sharding_helper",
    ],
)

cc_library(
    name = "canonicalize_all_gather_for_cse",
    srcs = ["canonicalize_all_gather_for_cse.cc"],
    hdrs = ["canonicalize_all_gather_for_cse.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
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
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "schedule_aware_collective_ops_cse",
    srcs = ["schedule_aware_collective_ops_cse.cc"],
    hdrs = ["schedule_aware_collective_ops_cse.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "spmd_prepare_test",
    srcs = ["spmd_prepare_test.cc"],
    deps = [
        ":spmd_prepare",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "spmd_prepare",
    srcs = ["spmd_prepare.cc"],
    hdrs = ["spmd_prepare.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:pattern_matcher",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "stateful_rng_spmd_partitioner",
    srcs = ["stateful_rng_spmd_partitioner.cc"],
    hdrs = ["stateful_rng_spmd_partitioner.h"],
    deps = [
        ":spmd_partitioner",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "stateful_rng_spmd_partitioner_test",
    srcs = ["stateful_rng_spmd_partitioner_test.cc"],
    deps = [
        ":stateful_rng_spmd_partitioner",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/expanders:rng_expander",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_verifier",
        "//xla/service:sharding_propagation",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "collective_permute_motion",
    srcs = ["collective_permute_motion.cc"],
    hdrs = ["collective_permute_motion.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_permute_motion_test",
    srcs = ["collective_permute_motion_test.cc"],
    deps = [
        ":collective_permute_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "partition_assignment_test",
    srcs = ["partition_assignment_test.cc"],
    deps = [
        ":partition_assignment",
        "//xla:xla_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "whole_graph_manual_pass_test",
    srcs = ["whole_graph_manual_pass_test.cc"],
    deps = [
        ":whole_graph_manual_pass",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
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
