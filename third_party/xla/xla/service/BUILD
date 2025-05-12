# Description:
#   XLA service implementation.

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_internal",
    "xla_py_proto_library",
)
load("//xla/service:xla_compile.bzl", "xla_aot_compile_cpu", "xla_aot_compile_gpu", "xla_aot_compile_gpu_runtime_autotuning")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_libtpu", "internal_visibility", "tsl_copts")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable", "internal_hlo_deps")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

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

tf_proto_library(
    name = "hlo_proto",
    srcs = ["hlo.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//xla:xla_data_proto",
        "//xla/service:metrics_proto",
    ] + if_google(["@com_google_protobuf//:any"]),
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "hlo_pyclif",
    proto_lib = ":hlo_proto",
    proto_srcfile = "hlo.proto",
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hlo_profile_printer_data",
    srcs = ["hlo_profile_printer_data.proto"],
)

tf_proto_library(
    name = "hlo_execution_profile_data",
    srcs = ["hlo_execution_profile_data.proto"],
    protodeps = [":hlo_profile_printer_data"],
)

tf_proto_library(
    name = "metrics_proto",
    srcs = ["metrics.proto"],
    make_default_target_header_only = True,
    protodeps = if_google([
        "@com_google_protobuf//:any",
        "@com_google_protobuf//:duration",
        "@com_google_protobuf//:timestamp",
    ]),
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "metrics_pb2",
    deps = [":metrics_proto"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "collective_opt_utils",
    srcs = ["collective_opt_utils.cc"],
    hdrs = ["collective_opt_utils.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "all_reduce_key",
    srcs = ["all_reduce_key.cc"],
    hdrs = ["all_reduce_key.h"],
    deps = [
        ":hlo_domain_map",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "all_reduce_promotion",
    srcs = ["all_reduce_promotion.cc"],
    hdrs = ["all_reduce_promotion.h"],
    deps = [
        ":change_op_data_type",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "all_reduce_promotion_test",
    srcs = ["all_reduce_promotion_test.cc"],
    deps = [
        ":all_reduce_promotion",
        ":pattern_matcher",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_reduce_reassociate",
    srcs = ["all_reduce_reassociate.cc"],
    hdrs = ["all_reduce_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":pattern_matcher",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "all_reduce_reassociate_test",
    srcs = ["all_reduce_reassociate_test.cc"],
    deps = [
        ":all_reduce_reassociate",
        ":pattern_matcher",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "float_support",
    srcs = ["float_support.cc"],
    hdrs = ["float_support.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "source_target_pairs",
    hdrs = ["source_target_pairs.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "source_target_pairs_test",
    srcs = ["source_target_pairs_test.cc"],
    deps = [
        ":source_target_pairs",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_permute_cycle",
    srcs = ["collective_permute_cycle.cc"],
    hdrs = ["collective_permute_cycle.h"],
    deps = [
        ":source_target_pairs",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "collective_permute_cycle_test",
    srcs = ["collective_permute_cycle_test.cc"],
    deps = [
        ":collective_permute_cycle",
        ":source_target_pairs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_permute_decomposer",
    srcs = ["collective_permute_decomposer.cc"],
    hdrs = ["collective_permute_decomposer.h"],
    deps = [
        ":call_graph",
        ":collective_conflict_analysis",
        ":collective_ops_utils",
        ":collective_permute_cycle",
        ":pattern_matcher",
        ":source_target_pairs",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "collective_permute_decomposer_test",
    srcs = ["collective_permute_decomposer_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":collective_permute_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_permute_key",
    srcs = ["collective_permute_key.cc"],
    hdrs = ["collective_permute_key.h"],
    deps = [
        ":hlo_domain_map",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "collective_conflict_analysis",
    srcs = ["collective_conflict_analysis.cc"],
    hdrs = ["collective_conflict_analysis.h"],
    deps = [
        ":collective_ops_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "constant_value",
    srcs = ["constant_value.cc"],
    hdrs = ["constant_value.h"],
    deps = [
        "//xla:literal",
        "//xla:util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "constant_value_test",
    srcs = ["constant_value_test.cc"],
    deps = [
        ":constant_value",
        "//xla:literal_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "value_range",
    srcs = ["value_range.cc"],
    hdrs = ["value_range.h"],
    deps = [
        ":constant_value",
        ":hlo_value",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "value_range_test",
    srcs = ["value_range_test.cc"],
    deps = [
        ":constant_value",
        ":hlo_module_config",
        ":value_range",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "collective_pipeliner_utils",
    hdrs = ["collective_pipeliner_utils.h"],
    deps = [],
)

cc_library(
    name = "collective_pipeliner",
    srcs = ["collective_pipeliner.cc"],
    hdrs = ["collective_pipeliner.h"],
    deps = [
        ":collective_ops_utils",
        ":collective_pipeliner_utils",
        ":constant_value",
        ":scheduling_annotations_util",
        ":value_range",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/utils:hlo_query",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "collective_pipeliner_test",
    srcs = ["collective_pipeliner_test.cc"],
    deps = [
        ":collective_pipeliner",
        ":collective_pipeliner_utils",
        ":hlo_module_config",
        ":hlo_verifier",
        ":legalize_scheduling_annotations",
        ":memory_annotations_hdr",
        ":scheduling_annotations_util",
        "//xla:literal_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        ":hlo_graph_dumper",
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/runtime/large_hlo_snapshot_serialization:serialization",
        "//xla/tsl/lib/io:zlib_compression_options",
        "//xla/tsl/lib/io:zlib_outputbuffer",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ] + if_google([
        "@com_google_absl//absl/base:builddata",
    ]),
)

cc_library(
    name = "overload",
    hdrs = ["overload.h"],
)

xla_cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    # Required to create BuildData as stamp data is disabled by default in tests.
    stamp = 1,
    deps = [
        ":dump",
        ":hlo_module_config",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/runtime/large_hlo_snapshot_serialization:serialization",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "sharding_propagation",
    srcs = [
        "sharding_propagation.cc",
    ],
    hdrs = [
        "sharding_propagation.h",
    ],
    deps = [
        ":call_graph",
        ":custom_call_sharding_helper",
        ":dot_as_convolution_util",
        ":hlo_graph_dumper",
        ":memory_annotations_hdr",
        "//xla:array",
        "//xla:protobuf_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:sharding_op_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service/spmd:shard_barrier_partitioner",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "sharding_propagation_test",
    srcs = [
        "sharding_propagation_test.cc",
    ],
    deps = [
        ":sharding_propagation",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:hlo_constant_splitter",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "sharding_remover",
    srcs = [
        "sharding_remover.cc",
    ],
    hdrs = [
        "sharding_remover.h",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/spmd:shard_barrier_partitioner",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "sharding_remover_test",
    size = "small",
    srcs = [
        "sharding_remover_test.cc",
    ],
    deps = [
        ":sharding_remover",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dot_as_convolution_util",
    srcs = [
        "dot_as_convolution_util.cc",
    ],
    hdrs = [
        "dot_as_convolution_util.h",
    ],
    deps = [
        ":shape_inference",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "dynamic_parameter_binding_test",
    srcs = ["dynamic_parameter_binding_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "dynamic_update_slice_test",
    srcs = ["dynamic_update_slice_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/testlib:test",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "dfs_hlo_visitor_with_default_test",
    srcs = ["dfs_hlo_visitor_with_default_test.cc"],
    deps = [
        ":hlo_runner",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "pattern_matcher",
    hdrs = ["pattern_matcher.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:ptrvec",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
    ],
)

xla_cc_test(
    name = "pattern_matcher_test",
    srcs = ["pattern_matcher_test.cc"],
    deps = [
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "pattern_matcher_gmock_test",
    srcs = ["pattern_matcher_gmock_test.cc"],
    deps = [
        ":pattern_matcher",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "fuzzy_matcher",
    hdrs = ["fuzzy_matcher.h"],
    deps = [
        ":pattern_matcher",
        "//xla/hlo/ir:hlo",
    ],
)

xla_cc_test(
    name = "fuzzy_matcher_test",
    srcs = ["fuzzy_matcher_test.cc"],
    deps = [
        ":fuzzy_matcher",
        ":pattern_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_instruction_test",
    srcs = ["hlo_instruction_test.cc"],
    tags = ["not_run:arm"],
    deps = [
        ":hlo_proto_cc",
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_sharding_test",
    srcs = ["hlo_sharding_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "call_graph",
    srcs = ["call_graph.cc"],
    hdrs = ["call_graph.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "call_graph_test",
    srcs = ["call_graph_test.cc"],
    deps = [
        ":call_graph",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "call_inliner",
    srcs = ["call_inliner.cc"],
    hdrs = ["call_inliner.h"],
    deps = [
        ":call_graph",
        ":hlo_domain_isolator",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service/spmd/shardy:constants",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "call_inliner_test",
    size = "small",
    srcs = ["call_inliner_test.cc"],
    deps = [
        ":call_inliner",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "platform_util",
    srcs = ["platform_util.cc"],
    hdrs = ["platform_util.h"],
    deps = [
        ":compiler",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":compiler",
        ":computation_placer",
        ":platform_util",
        ":stream_pool",
        ":transfer_manager",
        "//xla:util",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/host:host_platform_id",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":allocation_tracker",
        ":backend",
        ":channel_tracker",
        ":compilation_cache",
        ":compiler",
        ":computation_layout",
        ":computation_placer",
        ":dump",
        ":dynamic_dimension_inference",
        ":dynamic_padder",
        ":executable",
        ":execution_tracker",
        ":hlo_execution_profile",
        ":hlo_module_config",
        ":hlo_module_util",
        ":hlo_proto_cc",
        ":hlo_proto_util",
        ":shaped_buffer",
        ":stream_pool",
        ":transfer_manager",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "local_service",
    srcs = ["local_service.cc"],
    hdrs = ["local_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":executable",
        ":hlo_module_config",
        ":local_service_utils",
        ":platform_util",
        ":service",
        ":shaped_buffer",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "local_service_utils",
    srcs = ["local_service_utils.cc"],
    hdrs = ["local_service_utils.h"],
    deps = [
        ":backend",
        ":hlo_module_config",
        ":hlo_module_util",
        ":hlo_proto_cc",
        ":service",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "latency_hiding_scheduler",
    srcs = ["latency_hiding_scheduler.cc"],
    hdrs = ["latency_hiding_scheduler.h"],
    deps = [
        ":dump",
        ":hlo_buffer",
        ":hlo_cost_analysis",
        ":hlo_value",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "latency_hiding_scheduler_test",
    srcs = ["latency_hiding_scheduler_test.cc"],
    deps = [
        ":hlo_cost_analysis",
        ":latency_hiding_scheduler",
        ":legalize_scheduling_annotations",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/collectives:async_collective_creator",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "p2p_schedule_preparation",
    srcs = ["p2p_schedule_preparation.cc"],
    hdrs = ["p2p_schedule_preparation.h"],
    deps = [
        ":collective_ops_utils",
        "//xla:util",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "p2p_schedule_preparation_test",
    srcs = ["p2p_schedule_preparation_test.cc"],
    deps = [
        ":p2p_schedule_preparation",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "profile_guided_latency_estimator",
    srcs = ["profile_guided_latency_estimator.cc"],
    hdrs = ["profile_guided_latency_estimator.h"],
    deps = [
        ":latency_hiding_scheduler",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
    ],
)

xla_cc_test(
    name = "profile_guided_latency_estimator_test",
    srcs = ["profile_guided_latency_estimator_test.cc"],
    deps = [
        ":latency_hiding_scheduler",
        ":profile_guided_latency_estimator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

cc_library(
    name = "compile_only_service",
    srcs = ["compile_only_service.cc"],
    hdrs = ["compile_only_service.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":dump",
        ":platform_util",
        ":service",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cpu_plugin",
    deps = [
        ":service",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/stream_executor/host:host_platform",
    ],
)

cc_library(
    name = "gpu_plugin_impl",
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":gpu_plugin_without_collectives",
    ] + if_gpu_is_configured([
        "//xla/backends/gpu/runtime:thunk_runtime_dependencies",
    ]),
)

cc_library(
    name = "gpu_plugin_without_collectives",
    compatible_with = get_compatible_with_portable(),
    deps = [
    ] + if_gpu_is_configured([
        ":service",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_transfer_manager",
    ]) + if_cuda_is_configured([
        "//xla/service/gpu:nvptx_compiler",
        "//xla/stream_executor/cuda:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu:amdgpu_compiler",
        "//xla/stream_executor/rocm:stream_executor_rocm",
    ]),
)

cc_library(
    name = "gpu_plugin_stub",
)

alias(
    name = "gpu_plugin_noncuda",
    actual = if_libtpu("gpu_plugin_stub", "gpu_plugin_impl"),
)

alias(
    name = "gpu_plugin",
    actual = if_cuda("gpu_plugin_impl", "gpu_plugin_noncuda"),
)

cc_library(
    name = "interpreter_plugin",
    deps = [
        ":service",
        "//xla/backends/interpreter:compiler",
        "//xla/backends/interpreter:interpreter_transfer_manager",
        "//xla/backends/interpreter:platform",
    ],
)

cc_library(
    name = "shaped_buffer",
    srcs = ["shaped_buffer.cc"],
    hdrs = ["shaped_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "shaped_buffer_test",
    srcs = ["shaped_buffer_test.cc"],
    deps = [
        ":cpu_plugin",
        ":platform_util",
        ":shaped_buffer",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = [
        "executable.h",
        "service_executable_run_options.h",
    ],
    deps = [
        ":buffer_assignment",
        ":computation_layout",
        ":dump",
        ":hlo_execution_profile",
        ":hlo_graph_dumper",
        ":hlo_module_config",
        ":hlo_proto_cc",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        ":stream_pool",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + internal_hlo_deps(),
)

xla_cc_test(
    name = "executable_test",
    srcs = ["executable_test.cc"],
    deps = [
        ":executable",
        ":hlo_execution_profile",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    visibility = internal_visibility(["//xla/internal:hwi_internal"]),
    deps = [
        ":buffer_assignment",
        ":buffer_value",
        ":computation_placer",
        ":executable",
        ":hlo_module_config",
        ":metrics_hook_interface",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_test(
    name = "compiler_test",
    srcs = ["compiler_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    deps = [
        ":compiler",
        "//xla:autotune_results_proto_cc",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_init",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "llvm_compiler",
    srcs = ["llvm_compiler.cc"],
    hdrs = ["llvm_compiler.h"],
    deps = [
        ":compiler",
        ":executable",
        ":stream_pool",
        "//xla:executable_run_options",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:denormal",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "transfer_manager",
    srcs = ["transfer_manager.cc"],
    hdrs = ["transfer_manager.h"],
    deps = [
        ":compiler",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "allocation_tracker",
    srcs = ["allocation_tracker.cc"],
    hdrs = ["allocation_tracker.h"],
    deps = [
        ":backend",
        ":shaped_buffer",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "execution_tracker",
    srcs = ["execution_tracker.cc"],
    hdrs = ["execution_tracker.h"],
    deps = [
        ":backend",
        ":stream_pool",
        "//xla:executable_run_options",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "channel_tracker",
    srcs = ["channel_tracker.cc"],
    hdrs = ["channel_tracker.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "name_uniquer",
    srcs = ["name_uniquer.cc"],
    hdrs = ["name_uniquer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "name_uniquer_test",
    srcs = ["name_uniquer_test.cc"],
    deps = [
        ":name_uniquer",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "buffer_assignment",
    srcs = [
        "buffer_assignment.cc",
    ],
    hdrs = [
        "buffer_assignment.h",
    ],
    deps = [
        ":buffer_assignment_proto_cc",
        ":buffer_value",
        ":call_graph",
        ":hlo_buffer",
        ":hlo_proto_cc",
        ":hlo_value",
        ":logical_buffer",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service/heap_simulator",
        "//xla/service/memory_space_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
    ],
)

xla_cc_test(
    name = "buffer_assignment_test",
    srcs = ["buffer_assignment_test.cc"],
    deps = [
        ":buffer_assignment",
        ":buffer_value",
        ":call_graph",
        ":copy_insertion",
        ":cpu_plugin",
        ":hlo_buffer",
        ":hlo_proto_cc",
        ":hlo_value",
        ":logical_buffer",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/service/memory_space_assignment",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_group_test",
    srcs = ["hlo_module_group_test.cc"],
    # TODO(b/148211710) Test fails in OSS.
    tags = ["no_oss"],
    deps = [
        ":hlo_module_group_metadata",
        ":hlo_proto_cc",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "hlo_module_group_metadata",
    srcs = ["hlo_module_group_metadata.cc"],
    hdrs = ["hlo_module_group_metadata.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_module_util",
    srcs = ["hlo_module_util.cc"],
    hdrs = ["hlo_module_util.h"],
    deps = [
        ":compiler",
        ":computation_layout",
        ":hlo_module_config",
        ":hlo_verifier",
        "//xla:debug_options_flags",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_module_group_util",
    srcs = ["hlo_module_group_util.cc"],
    hdrs = ["hlo_module_group_util.h"],
    deps = [
        ":hlo_module_group_metadata",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_schedule_test",
    srcs = ["hlo_schedule_test.cc"],
    deps = [
        ":buffer_value",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "hlo_input_output_alias_config_test",
    srcs = ["hlo_input_output_alias_config_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "fusion_queue",
    hdrs = ["fusion_queue.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":fusion_queue",
        ":hlo_graph_dumper",
        ":hlo_module_config",
        ":pattern_matcher",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

xla_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    deps = [
        ":instruction_fusion",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_creation_utils",
    srcs = ["hlo_creation_utils.cc"],
    hdrs = [
        "hlo_creation_utils.h",
    ],
    deps = [
        ":hlo_module_config",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "fusion_node_indexing_evaluation",
    srcs = ["fusion_node_indexing_evaluation.cc"],
    hdrs = ["fusion_node_indexing_evaluation.h"],
    deps = [
        ":elemental_ir_emitter",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "fusion_node_indexing_evaluation_test",
    srcs = ["fusion_node_indexing_evaluation_test.cc"],
    deps = [
        ":fusion_node_indexing_evaluation",
        ":instruction_fusion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "hlo_creation_utils_test",
    srcs = ["hlo_creation_utils_test.cc"],
    deps = [
        ":hlo_creation_utils",
        ":pattern_matcher",
        "//xla:array2d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "batchnorm_expander",
    srcs = ["batchnorm_expander.cc"],
    hdrs = ["batchnorm_expander.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gather_expander",
    srcs = ["gather_expander.cc"],
    hdrs = ["gather_expander.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo_creation_utils",
        ":while_util",
        "//xla:literal_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scatter_utils",
    srcs = ["scatter_utils.cc"],
    hdrs = ["scatter_utils.h"],
    deps = [
        ":call_inliner",
        ":hlo_creation_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "scatter_expander",
    srcs = ["scatter_expander.cc"],
    hdrs = ["scatter_expander.h"],
    deps = [
        ":gather_scatter_utils",
        ":hlo_creation_utils",
        ":scatter_utils",
        ":while_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scatter_determinism_expander",
    srcs = ["scatter_determinism_expander.cc"],
    hdrs = ["scatter_determinism_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":scatter_utils",
        "//xla:array",
        "//xla:array2d",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "scatter_expander_test",
    srcs = ["scatter_expander_test.cc"],
    deps = [
        ":scatter_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_test(
    name = "scatter_determinism_expander_test",
    srcs = ["scatter_determinism_expander_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":scatter_determinism_expander",
        "//xla:literal",
        "//xla/hlo/testlib:test",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "triangular_solve_expander",
    srcs = ["triangular_solve_expander.cc"],
    hdrs = ["triangular_solve_expander.h"],
    deps = [
        ":hlo_creation_utils",
        ":hlo_module_config",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:slicing",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "triangular_solve_expander_test",
    size = "medium",
    srcs = ["triangular_solve_expander_test.cc"],
    shard_count = 12,
    deps = [
        ":triangular_solve_expander",
        "//xla:array2d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:reference_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "batchnorm_expander_test",
    size = "small",
    srcs = ["batchnorm_expander_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":batchnorm_expander",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
    ],
)

xla_test(
    name = "algebraic_simplifier_overflow_test",
    srcs = ["algebraic_simplifier_overflow_test.cc"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        "//xla:error_spec",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "collective_combiner_utils",
    hdrs = ["collective_combiner_utils.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "collective_decomposer_utils",
    srcs = ["collective_decomposer_utils.cc"],
    hdrs = ["collective_decomposer_utils.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_module_config",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "reduce_scatter_combiner",
    srcs = ["reduce_scatter_combiner.cc"],
    hdrs = ["reduce_scatter_combiner.h"],
    deps = [
        ":all_reduce_key",
        ":collective_combiner_utils",
        ":collective_ops_utils",
        ":hlo_domain_map",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "reduce_scatter_combiner_test",
    srcs = ["reduce_scatter_combiner_test.cc"],
    deps = [
        ":reduce_scatter_combiner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "all_reduce_simplifier",
    srcs = ["all_reduce_simplifier.cc"],
    hdrs = ["all_reduce_simplifier.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_module_config",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_replication_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "all_reduce_simplifier_test",
    srcs = ["all_reduce_simplifier_test.cc"],
    deps = [
        ":all_reduce_simplifier",
        ":hlo_module_config",
        ":pattern_matcher",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_gather_simplifier",
    srcs = ["all_gather_simplifier.cc"],
    hdrs = ["all_gather_simplifier.h"],
    deps = [
        ":collective_opt_utils",
        ":hlo_module_config",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "all_gather_simplifier_test",
    srcs = ["all_gather_simplifier_test.cc"],
    deps = [
        ":all_gather_simplifier",
        ":pattern_matcher",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "reduce_scatter_decomposer",
    srcs = ["reduce_scatter_decomposer.cc"],
    hdrs = ["reduce_scatter_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        ":hlo_module_config",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "reduce_scatter_decomposer_test",
    srcs = ["reduce_scatter_decomposer_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":reduce_scatter_decomposer",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "reduce_scatter_reassociate",
    srcs = ["reduce_scatter_reassociate.cc"],
    hdrs = ["reduce_scatter_reassociate.h"],
    deps = [
        ":all_reduce_key",
        ":collective_ops_utils",
        ":hlo_domain_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "reduce_scatter_reassociate_test",
    srcs = ["reduce_scatter_reassociate_test.cc"],
    deps = [
        ":reduce_scatter_reassociate",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "gather_expander_test",
    srcs = ["gather_expander_test.cc"],
    deps = [
        ":gather_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_query",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "conditional_simplifier",
    srcs = ["conditional_simplifier.cc"],
    hdrs = ["conditional_simplifier.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "conditional_simplifier_test",
    srcs = ["conditional_simplifier_test.cc"],
    deps = [
        ":conditional_simplifier",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "conditional_code_motion",
    srcs = ["conditional_code_motion.cc"],
    hdrs = ["conditional_code_motion.h"],
    deps = [
        ":hlo_cse",
        ":hlo_verifier",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "conditional_code_motion_test",
    srcs = ["conditional_code_motion_test.cc"],
    deps = [
        ":conditional_code_motion",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "space_to_batch_converter",
    srcs = ["space_to_batch_converter.cc"],
    hdrs = ["space_to_batch_converter.h"],
    deps = [
        ":hlo_creation_utils",
        ":pattern_matcher",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/lib/core:bitmap",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "space_to_batch_converter_test",
    size = "small",
    srcs = ["space_to_batch_converter_test.cc"],
    deps = [
        ":space_to_batch_converter",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "scan_loop_accumulator_input_unification",
    srcs = ["scan_loop_accumulator_input_unification.cc"],
    hdrs = ["scan_loop_accumulator_input_unification.h"],
    deps = [
        ":pattern_matcher",
        ":while_loop_simplifier",
        ":while_loop_unroller",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "scan_loop_accumulator_input_unification_test",
    srcs = ["scan_loop_accumulator_input_unification_test.cc"],
    deps = [
        ":copy_insertion",
        ":scan_loop_accumulator_input_unification",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_unstacker",
    srcs = ["hlo_unstacker.cc"],
    hdrs = ["hlo_unstacker.h"],
    deps = [
        ":hlo_creation_utils",
        ":pattern_matcher",
        ":tuple_util",
        ":while_loop_unroller",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_unstacker_test",
    srcs = ["hlo_unstacker_test.cc"],
    tags = if_google(["requires-net:external"]),
    deps = [
        ":hlo_unstacker",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_unroller",
    srcs = ["while_loop_unroller.cc"],
    hdrs = ["while_loop_unroller.h"],
    deps = [
        ":call_inliner",
        ":collective_ops_utils",
        ":constant_value",
        ":hlo_creation_utils",
        ":hlo_cse",
        ":pattern_matcher",
        ":scheduling_annotations_util",
        ":value_range",
        ":while_loop_constant_sinking",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "while_loop_unroller_test",
    srcs = ["while_loop_unroller_test.cc"],
    deps = [
        ":scheduling_annotations_util",
        ":while_loop_unroller",
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "while_loop_simplifier",
    srcs = ["while_loop_simplifier.cc"],
    hdrs = ["while_loop_simplifier.h"],
    deps = [
        ":call_inliner",
        ":hlo_creation_utils",
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:union_find",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_simplifier_test",
    srcs = ["while_loop_simplifier_test.cc"],
    deps = [
        ":while_loop_simplifier",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "all_to_all_decomposer",
    srcs = ["all_to_all_decomposer.cc"],
    hdrs = ["all_to_all_decomposer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "all_gather_decomposer",
    srcs = ["all_gather_decomposer.cc"],
    hdrs = ["all_gather_decomposer.h"],
    deps = [
        ":collective_decomposer_utils",
        ":collective_ops_utils",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "all_gather_decomposer_test",
    srcs = ["all_gather_decomposer_test.cc"],
    deps = [
        ":all_gather_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dynamic_window_utils",
    srcs = ["dynamic_window_utils.cc"],
    hdrs = ["dynamic_window_utils.h"],
    deps = [
        ":shape_inference",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "dynamic_dimension_inference",
    srcs = ["dynamic_dimension_inference.cc"],
    hdrs = ["dynamic_dimension_inference.h"],
    deps = [
        ":call_inliner",
        ":dynamic_window_utils",
        ":hlo_creation_utils",
        ":hlo_value",
        ":tuple_util",
        ":while_util",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dynamic_padder",
    srcs = ["dynamic_padder.cc"],
    hdrs = ["dynamic_padder.h"],
    deps = [
        ":call_graph",
        ":dynamic_dimension_inference",
        ":dynamic_window_utils",
        ":hlo_creation_utils",
        ":pattern_matcher",
        ":shape_inference",
        ":tuple_util",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/tsl/lib/monitoring:gauge",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "dynamic_padder_test",
    srcs = ["dynamic_padder_test.cc"],
    tags = ["test_xla_cpu_no_thunks"],
    deps = [
        ":dynamic_dimension_inference",
        ":dynamic_padder",
        ":pattern_matcher",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:dynamic_dimension_simplifier",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:client_library_test_base",
        "//xla/tests:hlo_test_base",
        "//xla/tests:llvm_irgen_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "dynamic_dimension_inference_test",
    srcs = ["dynamic_dimension_inference_test.cc"],
    deps = [
        ":dynamic_dimension_inference",
        ":hlo_runner",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "computation_placer",
    srcs = ["computation_placer.cc"],
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "//xla:array2d",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/stream_executor/sycl:sycl_platform_id",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ] + if_libtpu([":tpu_computation_placer"]),
    alwayslink = True,  # Contains per-platform computation placer registration
)

cc_library(
    name = "computation_placer_hdr",
    hdrs = ["computation_placer.h"],
    deps = [
        ":global_device_id",
        "//xla:array2d",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tpu_computation_placer",
    srcs = ["tpu_computation_placer.cc"],
    hdrs = ["tpu_computation_placer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":computation_placer_hdr",
        "//xla/stream_executor/tpu:status_helper",
        "//xla/stream_executor/tpu:tpu_api",
        "//xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//xla/stream_executor/tpu:tpu_platform_hdr",
        "//xla/stream_executor/tpu:tpu_platform_id",
        "//xla/stream_executor/tpu:tpu_topology_external",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,  # Contains TPU computation placer registration
)

cc_library(
    name = "human_readable_profile_builder",
    srcs = ["human_readable_profile_builder.cc"],
    hdrs = ["human_readable_profile_builder.h"],
    deps = [
        "//xla:metric_table_report",
        "//xla:types",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
    ],
)

cc_library(
    name = "generic_transfer_manager",
    srcs = ["generic_transfer_manager.cc"],
    hdrs = ["generic_transfer_manager.h"],
    deps = [
        ":shaped_buffer",
        ":transfer_manager",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

xla_cc_test(
    name = "generic_transfer_manager_test",
    srcs = ["generic_transfer_manager_test.cc"],
    deps = [
        ":generic_transfer_manager",
        ":shaped_buffer",
        ":transfer_manager",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:types",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_cost_analysis",
    srcs = ["hlo_cost_analysis.cc"],
    hdrs = ["hlo_cost_analysis.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/lib/gtl:map_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "hlo_cost_analysis_test",
    srcs = ["hlo_cost_analysis_test.cc"],
    deps = [
        ":cpu_plugin",
        ":hlo_cost_analysis",
        ":local_service",
        ":service",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/builder:padding",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_execution_profile",
    srcs = ["hlo_execution_profile.cc"],
    hdrs = ["hlo_execution_profile.h"],
    deps = [
        ":hlo_cost_analysis",
        ":hlo_execution_profile_data_cc",
        ":hlo_profile_printer",
        ":human_readable_profile_builder",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

xla_cc_test(
    name = "hlo_execution_profile_test",
    srcs = ["hlo_execution_profile_test.cc"],
    tags = ["no_mac_arm64"],
    deps = [
        ":cpu_plugin",
        ":hlo_cost_analysis",
        ":hlo_execution_profile",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_computation_test",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_test",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":buffer_value",
        ":computation_placer_hdr",
        ":hlo_module_config",
        ":test_compilation_environment_proto_cc",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "hlo_module_metadata_test",
    srcs = ["hlo_module_metadata_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_value",
    srcs = ["buffer_value.cc"],
    hdrs = ["buffer_value.h"],
    deps = [
        ":hlo_proto_cc",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "buffer_value_containers",
    hdrs = ["buffer_value_containers.h"],
    deps = [
        ":buffer_value",
        ":logical_buffer",
        "//xla/tsl/lib/gtl:compactptrset",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "logical_buffer",
    srcs = ["logical_buffer.cc"],
    hdrs = ["logical_buffer.h"],
    deps = [
        ":buffer_value",
        ":hlo_proto_cc",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/lib/gtl:int_type",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_value",
    srcs = ["hlo_value.cc"],
    hdrs = ["hlo_value.h"],
    deps = [
        ":buffer_value",
        "//xla:lazy",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_phi_graph",
    srcs = ["hlo_phi_graph.cc"],
    hdrs = ["hlo_phi_graph.h"],
    deps = [
        ":hlo_value",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "hlo_phi_graph_test",
    srcs = ["hlo_phi_graph_test.cc"],
    deps = [
        ":hlo_graph_dumper",
        ":hlo_phi_graph",
        "//xla:literal_util",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_buffer",
    srcs = ["hlo_buffer.cc"],
    hdrs = ["hlo_buffer.h"],
    deps = [
        ":hlo_value",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "compilation_cache",
    srcs = ["compilation_cache.cc"],
    hdrs = ["compilation_cache.h"],
    deps = [
        ":executable",
        ":hlo_module_config",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "layout_assignment",
    srcs = [
        "layout_assignment.cc",
    ],
    hdrs = [
        "layout_assignment.h",
    ],
    deps = [
        ":call_graph",
        ":computation_layout",
        ":hlo_graph_dumper",
        ":logical_buffer",
        "//xla:permutation_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "copy_insertion",
    srcs = [
        "copy_insertion.cc",
        "copy_removal.cc",
    ],
    hdrs = [
        "compile_time_cap.h",
        "copy_insertion.h",
        "copy_removal.h",
    ],
    deps = [
        ":call_graph",
        ":dump",
        ":hlo_buffer",
        ":hlo_graph_dumper",
        ":hlo_value",
        "//xla:frontend_attributes",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:ptrvec",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "loop_schedule_linearizer",
    srcs = ["loop_schedule_linearizer.cc"],
    hdrs = ["loop_schedule_linearizer.h"],
    deps = [
        ":hlo_graph_dumper",
        ":hlo_value",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/graphcycles",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "copy_insertion_test",
    srcs = ["copy_insertion_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo_graph_dumper",
        ":hlo_module_config",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/utils:hlo_query",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "loop_schedule_linearizer_test",
    srcs = ["loop_schedule_linearizer_test.cc"],
    deps = [
        ":copy_insertion",
        ":hlo_graph_dumper",
        ":loop_schedule_linearizer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_module_dce",
    srcs = ["hlo_module_dce.cc"],
    hdrs = ["hlo_module_dce.h"],
    deps = [
        ":while_loop_simplifier",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/analysis:hlo_liveness_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_verifier",
    srcs = ["hlo_verifier.cc"],
    hdrs = ["hlo_verifier.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_module_config",
        ":shape_inference",
        "//xla:comparison_util",
        "//xla:permutation_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
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

xla_cc_test(
    name = "hlo_verifier_test",
    srcs = ["hlo_verifier_test.cc"],
    deps = [
        ":hlo_module_config",
        ":hlo_verifier",
        ":layout_assignment",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "cpu_gpu_shape_verifier",
    srcs = ["cpu_gpu_shape_verifier.cc"],
    hdrs = ["cpu_gpu_shape_verifier.h"],
    deps = [
        ":collective_ops_utils",
        ":hlo_verifier",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "cpu_gpu_shape_verifier_test",
    srcs = ["cpu_gpu_shape_verifier_test.cc"],
    deps = [
        ":cpu_gpu_shape_verifier",
        ":hlo_module_config",
        ":hlo_verifier",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_dce_test",
    srcs = ["hlo_module_dce_test.cc"],
    deps = [
        ":hlo_module_dce",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

xla_cc_test(
    name = "layout_assignment_test",
    srcs = ["layout_assignment_test.cc"],
    deps = [
        ":computation_layout",
        ":layout_assignment",
        ":logical_buffer",
        ":pattern_matcher",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_cse",
    srcs = ["hlo_cse.cc"],
    hdrs = ["hlo_cse.h"],
    deps = [
        ":hlo_domain_map",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "hlo_cse_test",
    srcs = ["hlo_cse_test.cc"],
    deps = [
        ":hlo_cse",
        ":pattern_matcher",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "hlo_domain_map",
    srcs = ["hlo_domain_map.cc"],
    hdrs = ["hlo_domain_map.h"],
    deps = [
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_domain_verifier",
    srcs = ["hlo_domain_verifier.cc"],
    hdrs = ["hlo_domain_verifier.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_graph_dumper",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hlo_domain_isolator",
    srcs = ["hlo_domain_isolator.cc"],
    hdrs = ["hlo_domain_isolator.h"],
    deps = [
        ":hlo_domain_remover",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_domain_remover",
    srcs = ["hlo_domain_remover.cc"],
    hdrs = ["hlo_domain_remover.h"],
    deps = [
        ":hlo_domain_map",
        ":hlo_domain_verifier",
        ":hlo_graph_dumper",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_domain_test",
    srcs = ["hlo_domain_test.cc"],
    deps = [
        ":call_inliner",
        ":hlo_domain_isolator",
        ":hlo_domain_remover",
        ":hlo_domain_verifier",
        ":sharding_propagation",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "maybe_owning_device_memory",
    srcs = [
        "maybe_owning_device_memory.cc",
    ],
    hdrs = [
        "maybe_owning_device_memory.h",
    ],
    deps = [
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "float8_fnuz_ir_emitter",
    srcs = [
        "float8_fnuz_ir_emitter.cc",
    ],
    hdrs = [
        "float8_fnuz_ir_emitter.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":algorithm_util",
        ":float8_fnuz_ir_emitter",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:ir_builder_mixin",
        "//xla/service/llvm_ir:llvm_loop",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

xla_test(
    name = "elemental_ir_emitter_test",
    srcs = ["elemental_ir_emitter_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    # TODO(b/332870133): Enable when it passes on H100.
    disabled_backends = ["gpu_h100"],
    tags = [
        "no_windows",  # TODO(b/152037541)
    ],
    deps = [
        ":elemental_ir_emitter",
        ":hlo_module_config",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_module_config",
    srcs = ["hlo_module_config.cc"],
    hdrs = ["hlo_module_config.h"],
    deps = [
        ":computation_layout",
        ":computation_placer",
        ":hlo_proto_cc",
        ":sharding_config",
        "//xla:debug_options_flags",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_config_test",
    srcs = ["hlo_module_config_test.cc"],
    deps = [
        ":hlo_module_config",
        "//xla:xla_proto_cc",
        "//xla/tests:test_utils",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "computation_layout",
    srcs = ["computation_layout.cc"],
    hdrs = ["computation_layout.h"],
    deps = [
        "//xla:printer",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_graph_dumper",
    srcs = ["hlo_graph_dumper.cc"],
    hdrs = ["hlo_graph_dumper.h"],
    deps = [
        ":pattern_matcher",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:dnn",
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/lib/io:zlib_compression_options",
        "//xla/tsl/lib/io:zlib_outputbuffer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:base64",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "hlo_graph_dumper_test",
    srcs = ["hlo_graph_dumper_test.cc"],
    deps = [
        ":hlo_graph_dumper",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "transpose_folding",
    srcs = ["transpose_folding.cc"],
    hdrs = ["transpose_folding.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "transpose_folding_test",
    srcs = ["transpose_folding_test.cc"],
    deps = [
        ":shape_inference",
        ":transpose_folding",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "stream_pool",
    srcs = ["stream_pool.cc"],
    hdrs = ["stream_pool.h"],
    deps = [
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "stream_pool_test",
    srcs = ["stream_pool_test.cc"],
    deps = [
        ":stream_pool",
        "//xla/hlo/testlib:test_helpers",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/host:host_platform",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_proto_util",
    srcs = ["hlo_proto_util.cc"],
    hdrs = ["hlo_proto_util.h"],
    deps = [
        ":buffer_assignment",
        ":hlo_proto_cc",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
    ],
)

xla_cc_test(
    name = "hlo_proto_util_test",
    srcs = ["hlo_proto_util_test.cc"],
    deps = [
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_runner_interface",
    srcs = ["hlo_runner_interface.cc"],
    hdrs = ["hlo_runner_interface.h"],
    deps = [
        ":computation_placer",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_runner",
    srcs = ["hlo_runner.cc"],
    hdrs = ["hlo_runner.h"],
    deps = [
        ":backend",
        ":compiler",
        ":computation_layout",
        ":computation_placer",
        ":executable",
        ":hlo_module_util",
        ":hlo_runner_interface",
        ":maybe_owning_device_memory",
        ":shaped_buffer",
        ":transfer_manager",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "hlo_runner_pjrt",
    srcs = ["hlo_runner_pjrt.cc"],
    hdrs = ["hlo_runner_pjrt.h"],
    deps = [
        ":computation_layout",
        ":computation_placer_hdr",
        ":hlo_module_util",
        ":hlo_runner_interface",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "hlo_runner_pjrt_test",
    srcs = ["hlo_runner_pjrt_test.cc"],
    deps = [
        ":computation_placer_hdr",
        ":hlo_runner_interface",
        ":hlo_runner_pjrt",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/interpreter:interpreter_client",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "hlo_profile_printer",
    srcs = ["hlo_profile_printer.cc"],
    hdrs = ["hlo_profile_printer.h"],
    deps = [
        ":hlo_profile_printer_data_cc",
        ":human_readable_profile_builder",
        "//xla:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tuple_util",
    srcs = ["tuple_util.cc"],
    hdrs = ["tuple_util.h"],
    deps = [
        ":hlo_value",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "tuple_util_test",
    srcs = ["tuple_util_test.cc"],
    deps = [
        ":hlo_module_config",
        ":tuple_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "memory_annotations_hdr",
    hdrs = ["memory_annotations.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "host_offload_utils",
    srcs = ["host_offload_utils.cc"],
    hdrs = ["host_offload_utils.h"],
    deps = [
        ":call_graph",
        ":hlo_buffer",
        ":memory_annotations_hdr",
        ":pattern_matcher",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offload_utils_test",
    srcs = ["host_offload_utils_test.cc"],
    deps = [
        ":host_offload_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_util",
    srcs = ["while_util.cc"],
    hdrs = ["while_util.h"],
    deps = [
        ":call_inliner",
        ":hlo_creation_utils",
        ":pattern_matcher",
        ":tuple_util",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_util_test",
    srcs = ["while_util_test.cc"],
    deps = [
        ":while_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_all_reduce_code_motion",
    srcs = ["while_loop_all_reduce_code_motion.cc"],
    hdrs = ["while_loop_all_reduce_code_motion.h"],
    deps = [
        ":collective_ops_utils",
        "//xla:literal_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_replication_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/collectives:while_loop_all_reduce_code_motion_setup",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_all_reduce_code_motion_test",
    srcs = ["while_loop_all_reduce_code_motion_test.cc"],
    deps = [
        ":hlo_verifier",
        ":while_loop_all_reduce_code_motion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_concat_code_motion",
    srcs = ["while_loop_concat_code_motion.cc"],
    hdrs = ["while_loop_concat_code_motion.h"],
    deps = [
        ":while_loop_simplifier",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "while_loop_concat_code_motion_test",
    srcs = ["while_loop_concat_code_motion_test.cc"],
    deps = [
        ":hlo_verifier",
        ":while_loop_concat_code_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "while_loop_invariant_code_motion",
    srcs = ["while_loop_invariant_code_motion.cc"],
    hdrs = [
        "compile_time_cap.h",
        "while_loop_invariant_code_motion.h",
    ],
    deps = [
        ":while_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_invariant_code_motion_test",
    srcs = ["while_loop_invariant_code_motion_test.cc"],
    deps = [
        ":while_loop_invariant_code_motion",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_expensive_invariant_code_motion",
    srcs = ["while_loop_expensive_invariant_code_motion.cc"],
    hdrs = ["while_loop_expensive_invariant_code_motion.h"],
    deps = [
        ":while_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "while_loop_expensive_invariant_code_motion_test",
    srcs = ["while_loop_expensive_invariant_code_motion_test.cc"],
    deps = [
        ":while_loop_expensive_invariant_code_motion",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_constant_sinking",
    srcs = ["while_loop_constant_sinking.cc"],
    hdrs = ["while_loop_constant_sinking.h"],
    deps = [
        ":while_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_constant_sinking_test",
    srcs = ["while_loop_constant_sinking_test.cc"],
    deps = [
        ":while_loop_constant_sinking",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "while_loop_fusible_sinking",
    srcs = ["while_loop_fusible_sinking.cc"],
    hdrs = ["while_loop_fusible_sinking.h"],
    deps = [
        ":pattern_matcher",
        ":while_util",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_fusible_sinking_test",
    srcs = ["while_loop_fusible_sinking_test.cc"],
    deps = [
        ":while_loop_fusible_sinking",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "source_map_util",
    srcs = [],
    hdrs = ["source_map_util.h"],
    deps = [
        ":executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "map_inliner",
    srcs = ["map_inliner.cc"],
    hdrs = ["map_inliner.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "compilation_stats",
    srcs = ["compilation_stats.cc"],
    hdrs = ["compilation_stats.h"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "map_inliner_test",
    srcs = ["map_inliner_test.cc"],
    deps = [
        ":map_inliner",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "hlo_casting_utils_test",
    srcs = ["hlo_casting_utils_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "conditional_to_select",
    srcs = ["conditional_to_select.cc"],
    hdrs = ["conditional_to_select.h"],
    deps = [
        ":call_graph",
        ":call_inliner",
        ":hlo_creation_utils",
        "//xla:status_macros",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "conditional_to_select_test",
    srcs = ["conditional_to_select_test.cc"],
    deps = [
        ":conditional_to_select",
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "custom_call_target_registry",
    srcs = ["custom_call_target_registry.cc"],
    hdrs = ["custom_call_target_registry.h"],
    visibility = ["//visibility:public"],
)

xla_cc_test(
    name = "custom_call_target_registry_test",
    srcs = ["custom_call_target_registry_test.cc"],
    deps = [
        ":custom_call_status",
        ":custom_call_target_registry",
        "//xla/hlo/testlib:test",
        "@com_google_googletest//:gtest_main",
    ],
)

# Exposes the public interface only and hides internal details. Suitable for
# linking into a static library or binary.
cc_library(
    name = "custom_call_status",
    srcs = [
        "custom_call_status.cc",
        "custom_call_status_internal.h",
    ],
    hdrs = ["custom_call_status.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "custom_call_status_hdrs",
    srcs = [
        "custom_call_status.h",
        "custom_call_status_internal.h",
    ],
    visibility = internal_visibility([":friends"]),
)

filegroup(
    name = "custom_call_status_srcs",
    srcs = ["custom_call_status.cc"],
    visibility = internal_visibility([":friends"]),
)

# Internal version that exposes internal details and private interfaces. For
# use by other XLA code only.
cc_library(
    name = "custom_call_status_internal",
    hdrs = [
        "custom_call_status_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":__subpackages__",
        "//tensorflow/compiler/tf2xla:__pkg__",
        "//xla/pjrt:__subpackages__",
        "//xla/backends/cpu/runtime:__subpackages__",
        "//xla/backends/gpu/runtime:__subpackages__",
    ]),
    deps = [
        ":custom_call_status",
        "@com_google_absl//absl/strings",
    ],
)

# Public headers only, suitable for inclusion in a shared library of CustomCall
# target functions.
cc_library(
    name = "custom_call_status_public_headers",
    hdrs = ["custom_call_status.h"],
    visibility = ["//visibility:public"],
)

xla_cc_test(
    name = "custom_call_status_test",
    srcs = ["custom_call_status_test.cc"],
    deps = [
        ":custom_call_status_internal",
        ":custom_call_status_test_c_caller",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "custom_call_status_test_c_caller",
    testonly = True,
    srcs = ["custom_call_status_test_c_caller.c"],
    hdrs = ["custom_call_status_test_c_caller.h"],
    deps = [":custom_call_status"],
)

cc_library(
    name = "slow_operation_alarm",
    srcs = ["slow_operation_alarm.cc"],
    hdrs = ["slow_operation_alarm.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "slow_operation_alarm_test",
    srcs = ["slow_operation_alarm_test.cc"],
    deps = [
        ":slow_operation_alarm",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "collective_ops_utils",
    srcs = ["collective_ops_utils.cc"],
    hdrs = ["collective_ops_utils.h"],
    deps = [
        ":collective_permute_cycle",
        ":computation_placer",
        ":global_device_id",
        ":pattern_matcher",
        ":source_target_pairs",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "collective_ops_utils_test",
    srcs = ["collective_ops_utils_test.cc"],
    deps = [
        ":collective_ops_utils",
        ":collective_permute_cycle",
        ":computation_placer",
        ":global_device_id",
        ":hlo_module_config",
        "//xla:array2d",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "topk_rewriter",
    srcs = ["topk_rewriter.cc"],
    hdrs = ["topk_rewriter.h"],
    deps = [
        ":hlo_creation_utils",
        ":pattern_matcher",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "topk_rewriter_test",
    srcs = ["topk_rewriter_test.cc"],
    deps = [
        ":pattern_matcher",
        ":topk_rewriter",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "global_device_id",
    srcs = ["global_device_id.cc"],
    hdrs = ["global_device_id.h"],
    deps = [
        "//xla/runtime:device_id",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_debug_info_manager",
    srcs = [
        "xla_debug_info_manager.cc",
    ],
    hdrs = [
        "xla_debug_info_manager.h",
    ],
    deps = [
        ":hlo_proto_cc",
        ":hlo_proto_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "xla_debug_info_manager_test",
    srcs = ["xla_debug_info_manager_test.cc"],
    deps = [
        ":hlo_module_config",
        ":hlo_proto_cc",
        ":xla_debug_info_manager",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
    ],
)

xla_py_proto_library(
    name = "hlo_pb2",
    visibility = ["//visibility:public"],
    deps = [":hlo_proto"],
)

cc_library(
    name = "mapped_ptr_container_sorter",
    hdrs = ["mapped_ptr_container_sorter.h"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "mapped_ptr_container_sorter_test",
    srcs = ["mapped_ptr_container_sorter_test.cc"],
    deps = [
        ":mapped_ptr_container_sorter",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "lockable",
    hdrs = ["lockable.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "lockable_test",
    srcs = ["lockable_test.cc"],
    deps = [
        ":lockable",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "rendezvous",
    srcs = ["rendezvous.cc"],
    hdrs = ["rendezvous.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "rendezvous_test",
    srcs = ["rendezvous_test.cc"],
    deps = [
        ":rendezvous",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compilation_environments",
    srcs = ["compilation_environments.cc"],
    hdrs = ["compilation_environments.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",  # fixdeps: keep
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_google(["@com_google_protobuf//:any_cc_proto"]),
)

cc_library(
    name = "custom_call_sharding_helper",
    srcs = ["custom_call_sharding_helper.cc"],
    hdrs = ["custom_call_sharding_helper.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

tf_proto_library(
    name = "test_compilation_environment_proto",
    testonly = 1,
    srcs = ["test_compilation_environment.proto"],
)

xla_cc_test(
    name = "compilation_environments_test",
    srcs = ["compilation_environments_test.cc"],
    deps = [
        ":compilation_environments",
        ":test_compilation_environment_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "layout_normalization",
    srcs = ["layout_normalization.cc"],
    hdrs = ["layout_normalization.h"],
    deps = [
        ":hlo_creation_utils",
        ":shape_inference",
        "//xla:literal",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scatter_simplifier",
    srcs = ["scatter_simplifier.cc"],
    hdrs = ["scatter_simplifier.h"],
    deps = [
        ":call_inliner",
        ":gather_scatter_utils",
        ":hlo_creation_utils",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "scatter_simplifier_test",
    srcs = ["scatter_simplifier_test.cc"],
    deps = [
        ":scatter_simplifier",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "select_and_scatter_expander",
    srcs = ["select_and_scatter_expander.cc"],
    hdrs = ["select_and_scatter_expander.h"],
    deps = [
        ":call_inliner",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
    ],
)

xla_cc_test(
    name = "select_and_scatter_expander_test",
    srcs = ["select_and_scatter_expander_test.cc"],
    deps = [
        ":select_and_scatter_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "layout_normalization_test",
    srcs = [
        "layout_normalization_test.cc",
    ],
    deps = [
        ":layout_normalization",
        ":scatter_simplifier",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "change_op_data_type",
    srcs = ["change_op_data_type.cc"],
    hdrs = ["change_op_data_type.h"],
    copts = tsl_copts(),
    deps = [
        ":hlo_creation_utils",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/cpu:onednn_contraction_rewriter",
    ],
)

cc_library(
    name = "gather_scatter_utils",
    srcs = ["gather_scatter_utils.cc"],
    hdrs = ["gather_scatter_utils.h"],
    deps = [
        ":hlo_creation_utils",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "batched_gather_scatter_normalizer",
    srcs = ["batched_gather_scatter_normalizer.cc"],
    hdrs = ["batched_gather_scatter_normalizer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metrics_hook_interface",
    hdrs = ["metrics_hook_interface.h"],
    deps = [
        ":metrics_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "batched_gather_scatter_normalizer_test",
    srcs = ["batched_gather_scatter_normalizer_test.cc"],
    deps = [
        ":batched_gather_scatter_normalizer",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "change_op_data_type_test",
    srcs = ["change_op_data_type_test.cc"],
    deps = [
        ":change_op_data_type",
        ":pattern_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/types:span",
    ],
)

build_test(
    name = "xla_compile_build_test",
    targets = [
        ":xla_compile",
    ],
)

xla_cc_binary(
    name = "xla_compile",
    srcs = ["xla_compile_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpu_plugin",
        "//xla/tools:xla_compile_lib",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:types",
    ] + if_cuda_is_configured([
        # keep sorted
        "//xla/service/gpu:executable_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:nvptx_compiler",
        "//xla/service/gpu:nvptx_compiler_impl",
        "//xla/stream_executor/cuda:all_runtime",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/gpu:gpu_init",
    ]) + if_rocm_is_configured([
        # keep sorted
        "//xla/service/gpu:amdgpu_compiler",
        "//xla/service/gpu:amdgpu_compiler_impl",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/stream_executor/gpu:gpu_init",
        "//xla/stream_executor/rocm:all_runtime",
        "//xla/stream_executor/rocm:rocm_platform",
    ]) + xla_internal(["tools:xsymbol_repository"]),
)

# A simple test of xla_aot_compile which generates an output file from an mhlo file.
xla_aot_compile_cpu(
    name = "xla_aot_compile_test_cpu_executable",
    module = "xla_aot_compile_test.mlir",
)

xla_aot_compile_cpu(
    name = "xla_aot_compile_stablehlo_test_cpu_executable",
    module = "xla_aot_compile_stablehlo_test.mlir",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable",
    autotune_results = "xla_aot_compile_test_autotune_results.txtpb",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.txtpb",
    module = "xla_aot_compile_test.mlir",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable_hlo",
    autotune_results = "xla_aot_compile_test_autotune_results.txtpb",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.txtpb",
    module = "xla_aot_compile_test.hlo",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable_constant",
    autotune_results = "xla_aot_compile_test_autotune_results.txtpb",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.txtpb",
    module = "xla_aot_compile_test_constant.mlir",
)

xla_aot_compile_gpu(
    name = "xla_aot_compile_test_gpu_executable_convolution",
    autotune_results = "xla_aot_compile_test_autotune_results.txtpb",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.txtpb",
    module = "xla_aot_compile_test_convolution.mlir",
)

xla_aot_compile_gpu_runtime_autotuning(
    name = "xla_aot_compile_test_gpu_executable_convolution_runtime_autotuning",
    gpu_target_config = "xla_aot_compile_test_gpu_target_config.txtpb",
    module = "xla_aot_compile_test_convolution.mlir",
)

xla_cc_test(
    name = "xla_aot_compile_cpu_test",
    srcs = ["xla_aot_compile_cpu_test.cc"],
    data = [":xla_aot_compile_test_cpu_executable"],
    tags = [
        "no_oss",
        "notap",
    ],
    deps = [
        ":cpu_plugin",
        ":platform_util",
        ":shaped_buffer",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "xla_aot_compile_stablehlo_cpu_test",
    srcs = ["xla_aot_compile_stablehlo_cpu_test.cc"],
    data = [":xla_aot_compile_stablehlo_test_cpu_executable"],
    tags = ["not_run:arm"],  # b/341355246
    deps = [
        ":cpu_plugin",
        ":platform_util",
        ":shaped_buffer",
        "//xla:error_spec",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "xla_aot_compile_gpu_test",
    srcs = ["xla_aot_compile_gpu_test.cc"],
    data = [
        ":xla_aot_compile_test_gpu_executable",
        ":xla_aot_compile_test_gpu_executable_constant",
        ":xla_aot_compile_test_gpu_executable_convolution",
        ":xla_aot_compile_test_gpu_executable_convolution_runtime_autotuning",
        ":xla_aot_compile_test_gpu_executable_hlo",
    ],
    tags = [
        "cuda-only",
        "gpu",
        "no_oss",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false positives in msan.
        "requires-gpu-sm60-only",
    ],
    deps = [
        ":gpu_plugin_impl",
        ":platform_util",
        ":shaped_buffer",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

tf_proto_library(
    name = "buffer_assignment_proto",
    srcs = ["buffer_assignment.proto"],
    make_default_target_header_only = True,
)

cc_library(
    name = "export_hlo",
    hdrs = ["export_hlo.h"],
    deps = [
        "//xla:autotune_results_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_compilation_environment",
    srcs = ["gpu_compilation_environment.cc"],
    hdrs = ["gpu_compilation_environment.h"],
    deps = [
        ":compilation_environments",
        "//xla:parse_flags_from_env",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "gpu_compilation_environment_test",
    size = "small",
    srcs = ["gpu_compilation_environment_test.cc"],
    deps = [
        ":compilation_environments",
        ":gpu_compilation_environment",
        "//xla:parse_flags_from_env",
        "//xla:xla_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "symbol_repository",
    hdrs = ["symbol_repository.h"],
    deps = [
        ":compiler",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    deps = [],
)

tf_proto_library(
    name = "xla_compile_result_proto",
    srcs = ["xla_compile_result.proto"],
    make_default_target_header_only = True,
    protodeps = [
        ":hlo_proto",
        "//xla/tsl/protobuf:status_proto",
    ] + if_google(["@com_google_protobuf//:duration"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "algorithm_util",
    srcs = ["algorithm_util.cc"],
    hdrs = ["algorithm_util.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "propagate_original_value_test",
    srcs = ["propagate_original_value_test.cc"],
    deps = [
        ":instruction_fusion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "while_loop_pipeline_unroller",
    srcs = ["while_loop_pipeline_unroller.cc"],
    hdrs = ["while_loop_pipeline_unroller.h"],
    deps = [
        ":call_inliner",
        ":while_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "while_loop_pipeline_unroller_test",
    srcs = ["while_loop_pipeline_unroller_test.cc"],
    deps = [
        ":copy_insertion",
        ":while_loop_pipeline_unroller",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_utils",
    hdrs = ["collective_utils.h"],
)

cc_library(
    name = "legalize_scheduling_annotations",
    srcs = ["legalize_scheduling_annotations.cc"],
    hdrs = ["legalize_scheduling_annotations.h"],
    deps = [
        ":scheduling_annotations_util",
        "//xla:side_effect_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:ptrvec",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "legalize_scheduling_annotations_test",
    srcs = ["legalize_scheduling_annotations_test.cc"],
    deps = [
        ":legalize_scheduling_annotations",
        ":scheduling_annotations_util",
        "//xla:side_effect_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test_helpers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "sharding_config",
    srcs = ["sharding_config.cc"],
    hdrs = ["sharding_config.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo_sharding",
    ],
)

xla_cc_test(
    name = "sharding_config_test",
    srcs = ["sharding_config_test.cc"],
    deps = [
        ":sharding_config",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo_sharding",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "scheduling_annotations_util",
    srcs = ["scheduling_annotations_util.cc"],
    hdrs = ["scheduling_annotations_util.h"],
    deps = [
        ":collective_pipeliner_utils",
        "//xla:side_effect_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "scheduling_annotations_util_test",
    srcs = ["scheduling_annotations_util_test.cc"],
    deps = [
        ":scheduling_annotations_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

exports_files(["xla_aot_compile_test_gpu_target_config.txtpb"])
