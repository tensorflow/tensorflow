# Description:
#   Implementation of XLA’s HLO transformations.

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "tsl_copts")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
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
    name = "hlo_constant_splitter",
    srcs = ["simplifiers/hlo_constant_splitter.cc"],
    hdrs = ["simplifiers/hlo_constant_splitter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_constant_splitter_test",
    srcs = ["simplifiers/hlo_constant_splitter_test.cc"],
    deps = [
        ":hlo_constant_splitter",
        ":hlo_dce",
        "//xla:test",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "all_reduce_folder",
    srcs = ["simplifiers/all_reduce_folder.cc"],
    hdrs = ["simplifiers/all_reduce_folder.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:all_reduce_key",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "all_reduce_folder_test",
    srcs = ["simplifiers/all_reduce_folder_test.cc"],
    deps = [
        ":all_reduce_folder",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "broadcast_canonicalizer",
    srcs = ["simplifiers/broadcast_canonicalizer.cc"],
    hdrs = ["simplifiers/broadcast_canonicalizer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "broadcast_canonicalizer_test",
    srcs = ["simplifiers/broadcast_canonicalizer_test.cc"],
    deps = [
        ":broadcast_canonicalizer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "bfloat16_conversion_folding",
    srcs = ["simplifiers/bfloat16_conversion_folding.cc"],
    hdrs = ["simplifiers/bfloat16_conversion_folding.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:float_support",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "bfloat16_conversion_folding_test",
    srcs = ["simplifiers/bfloat16_conversion_folding_test.cc"],
    deps = [
        ":bfloat16_conversion_folding",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:float_support",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "float_normalization",
    srcs = ["simplifiers/float_normalization.cc"],
    hdrs = ["simplifiers/float_normalization.h"],
    deps = [
        ":hlo_dce",
        ":tuple_simplifier",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:float_support",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "float_normalization_test",
    srcs = ["simplifiers/float_normalization_test.cc"],
    deps = [
        ":float_normalization",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:float_support",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_verifier",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "bfloat16_propagation",
    srcs = ["bfloat16_propagation.cc"],
    hdrs = ["bfloat16_propagation.h"],
    deps = [
        ":hlo_dce",
        ":tuple_simplifier",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:float_support",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "bfloat16_propagation_test",
    srcs = ["bfloat16_propagation_test.cc"],
    deps = [
        ":bfloat16_propagation",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:float_support",
        "//xla/service:hlo_verifier",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "flatten_call_graph",
    srcs = ["simplifiers/flatten_call_graph.cc"],
    hdrs = ["simplifiers/flatten_call_graph.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "hlo_computation_deduplicator",
    srcs = ["simplifiers/hlo_computation_deduplicator.cc"],
    hdrs = ["simplifiers/hlo_computation_deduplicator.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_computation_deduplicator_test",
    size = "small",
    srcs = ["simplifiers/hlo_computation_deduplicator_test.cc"],
    deps = [
        ":hlo_computation_deduplicator",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "flatten_call_graph_test",
    srcs = ["simplifiers/flatten_call_graph_test.cc"],
    deps = [
        ":flatten_call_graph",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:call_graph",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_memory_scheduler",
    srcs = ["simplifiers/hlo_memory_scheduler.cc"],
    hdrs = ["simplifiers/hlo_memory_scheduler.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:buffer_value",
        "//xla/service:logical_buffer",
        "//xla/service/heap_simulator",
        "//xla/tsl/lib/gtl:map_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

xla_cc_test(
    name = "hlo_memory_scheduler_test",
    srcs = ["simplifiers/hlo_memory_scheduler_test.cc"],
    deps = [
        ":hlo_dce",
        ":hlo_memory_scheduler",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:buffer_value",
        "//xla/service:hlo_value",
        "//xla/service:logical_buffer",
        "//xla/service/heap_simulator",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "op_expander_pass",
    srcs = ["expanders/op_expander_pass.cc"],
    hdrs = ["expanders/op_expander_pass.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "optimization_barrier_expander",
    srcs = ["expanders/optimization_barrier_expander.cc"],
    hdrs = ["expanders/optimization_barrier_expander.h"],
    deps = [
        ":op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "comparison_expander",
    srcs = ["expanders/comparison_expander.cc"],
    hdrs = ["expanders/comparison_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cholesky_expander",
    srcs = ["expanders/cholesky_expander.cc"],
    hdrs = ["expanders/cholesky_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "qr_expander",
    srcs = ["expanders/qr_expander.cc"],
    hdrs = ["expanders/qr_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:qr",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "real_imag_expander",
    srcs = ["expanders/real_imag_expander.cc"],
    hdrs = ["expanders/real_imag_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "real_imag_expander_test",
    size = "small",
    srcs = ["expanders/real_imag_expander_test.cc"],
    deps = [
        ":real_imag_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["expanders/eigh_expander.cc"],
    hdrs = ["expanders/eigh_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "convolution_4d_expander",
    srcs = ["expanders/convolution_4d_expander.cc"],
    hdrs = ["expanders/convolution_4d_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_4d_expander_test",
    srcs = ["expanders/convolution_4d_expander_test.cc"],
    deps = [
        "convolution_4d_expander",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "convolution_pred_expander",
    srcs = ["expanders/convolution_pred_expander.cc"],
    hdrs = ["expanders/convolution_pred_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_pred_expander_test",
    srcs = ["expanders/convolution_pred_expander_test.cc"],
    deps = [
        ":convolution_pred_expander",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "algebraic_simplifier",
    srcs = ["simplifiers/algebraic_simplifier.cc"],
    hdrs = ["simplifiers/algebraic_simplifier.h"],
    copts = tsl_copts(),
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_instruction_utils",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:gather_scatter_utils",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_module_config",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["simplifiers/tree_reduction_rewriter.cc"],
    hdrs = ["simplifiers/tree_reduction_rewriter.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:padding",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "algebraic_simplifier_test",
    srcs = ["simplifiers/algebraic_simplifier_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":hlo_constant_folding",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_creation_utils",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:layout_assignment",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "simplify_fp_conversions",
    srcs = ["simplifiers/simplify_fp_conversions.cc"],
    hdrs = ["simplifiers/simplify_fp_conversions.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "simplify_fp_conversions_test",
    srcs = ["simplifiers/simplify_fp_conversions_test.cc"],
    deps = [
        ":simplify_fp_conversions",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["expanders/logistic_expander.cc"],
    hdrs = ["expanders/logistic_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "logistic_expander_test",
    srcs = ["expanders/logistic_expander_test.cc"],
    deps = [
        ":logistic_expander",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:dynamic_padder",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["expanders/bitcast_dtypes_expander.cc"],
    hdrs = ["expanders/bitcast_dtypes_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:broadcast",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "bitcast_dtypes_expander_test",
    srcs = ["expanders/bitcast_dtypes_expander_test.cc"],
    deps = [
        ":bitcast_dtypes_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "batch_dot_simplification",
    srcs = ["simplifiers/batch_dot_simplification.cc"],
    hdrs = ["simplifiers/batch_dot_simplification.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "batch_dot_simplification_test",
    srcs = ["simplifiers/batch_dot_simplification_test.cc"],
    deps = [
        ":batch_dot_simplification",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "convolution_group_converter",
    srcs = ["simplifiers/convolution_group_converter.cc"],
    hdrs = ["simplifiers/convolution_group_converter.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "convolution_group_converter_test",
    size = "small",
    srcs = ["simplifiers/convolution_group_converter_test.cc"],
    deps = [
        ":convolution_group_converter",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "while_loop_trip_count_annotator",
    srcs = ["while_loop_trip_count_annotator.cc"],
    hdrs = ["while_loop_trip_count_annotator.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "while_loop_trip_count_annotator_test",
    srcs = ["while_loop_trip_count_annotator_test.cc"],
    deps = [
        ":while_loop_trip_count_annotator",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "defuser",
    srcs = ["defuser.cc"],
    hdrs = ["defuser.h"],
    deps = [
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "defuser_test",
    srcs = ["defuser_test.cc"],
    deps = [
        ":defuser",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "despecializer_test",
    srcs = ["despecializer_test.cc"],
    deps = [
        ":despecializer",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["expanders/dot_decomposer.cc"],
    hdrs = ["expanders/dot_decomposer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
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
    name = "dot_decomposer_test",
    srcs = ["expanders/dot_decomposer_test.cc"],
    deps = [
        ":dot_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "dot_dimension_merger",
    srcs = ["simplifiers/dot_dimension_merger.cc"],
    hdrs = ["simplifiers/dot_dimension_merger.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "dot_dimension_merger_test",
    srcs = ["simplifiers/dot_dimension_merger_test.cc"],
    deps = [
        ":dot_dimension_merger",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "dot_merger",
    srcs = ["simplifiers/dot_merger.cc"],
    hdrs = ["simplifiers/dot_merger.h"],
    deps = [
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:shape_inference",
        "//xla/service/graphcycles",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "dot_merger_test",
    srcs = ["simplifiers/dot_merger_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":dot_merger",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "convert_mover",
    srcs = ["simplifiers/convert_mover.cc"],
    hdrs = ["simplifiers/convert_mover.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "convert_mover_test",
    srcs = ["simplifiers/convert_mover_test.cc"],
    deps = [
        ":convert_mover",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "tuple_simplifier",
    srcs = ["simplifiers/tuple_simplifier.cc"],
    hdrs = ["simplifiers/tuple_simplifier.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "tuple_simplifier_test",
    srcs = ["simplifiers/tuple_simplifier_test.cc"],
    deps = [
        ":tuple_simplifier",
        "//xla:shape_util",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "reshape_mover",
    srcs = ["simplifiers/reshape_mover.cc"],
    hdrs = ["simplifiers/reshape_mover.h"],
    deps = [
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "reshape_decomposer",
    srcs = ["expanders/reshape_decomposer.cc"],
    hdrs = ["expanders/reshape_decomposer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "reduce_decomposer",
    srcs = ["expanders/reduce_decomposer.cc"],
    hdrs = ["expanders/reduce_decomposer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "reduce_decomposer_test",
    srcs = ["expanders/reduce_decomposer_test.cc"],
    deps = [
        ":reduce_decomposer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reshape_decomposer_test",
    srcs = ["expanders/reshape_decomposer_test.cc"],
    deps = [
        ":reshape_decomposer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "dynamic_dimension_simplifier",
    srcs = ["simplifiers/dynamic_dimension_simplifier.cc"],
    hdrs = ["simplifiers/dynamic_dimension_simplifier.h"],
    deps = [
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "dynamic_dimension_simplifier_test",
    srcs = ["simplifiers/dynamic_dimension_simplifier_test.cc"],
    deps = [
        ":dynamic_dimension_simplifier",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

xla_cc_test(
    name = "reshape_mover_test",
    srcs = ["simplifiers/reshape_mover_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":reshape_mover",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_verifier",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "memory_space_propagation",
    srcs = ["memory_space_propagation.cc"],
    hdrs = ["memory_space_propagation.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "memory_space_propagation_test",
    srcs = ["memory_space_propagation_test.cc"],
    deps = [
        ":memory_space_propagation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_dce",
    srcs = ["simplifiers/hlo_dce.cc"],
    hdrs = ["simplifiers/hlo_dce.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_rematerialization",
    srcs = ["simplifiers/hlo_rematerialization.cc"],
    hdrs = ["simplifiers/hlo_rematerialization.h"],
    deps = [
        ":hlo_dce",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:call_graph",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:logical_buffer",
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
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_rematerialization_test_utils",
    testonly = 1,
    hdrs = ["simplifiers/hlo_rematerialization_test_utils.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "hlo_rematerialization_test_utils_test",
    srcs = ["simplifiers/hlo_rematerialization_test_utils_test.cc"],
    deps = [
        ":hlo_rematerialization_test_utils",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "hlo_rematerialization_test",
    srcs = ["simplifiers/hlo_rematerialization_test.cc"],
    deps = [
        ":hlo_memory_scheduler",
        ":hlo_rematerialization",
        ":hlo_rematerialization_test_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_cost_analysis",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "hlo_dce_test",
    srcs = ["simplifiers/hlo_dce_test.cc"],
    deps = [
        ":hlo_dce",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_constant_folding",
    srcs = ["simplifiers/hlo_constant_folding.cc"],
    hdrs = ["simplifiers/hlo_constant_folding.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:slow_operation_alarm",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "hlo_constant_folding_test",
    srcs = ["simplifiers/hlo_constant_folding_test.cc"],
    deps = [
        ":hlo_constant_folding",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

# copybara:uncomment_begin(google-only)
# xla_cc_test(
#     name = "hlo_constant_folding_peak_heap_test",
#     srcs = ["simplifiers/hlo_constant_folding_peak_heap_test.cc"],
#     deps = [
#         ":hlo_constant_folding",
#         "@com_google_googletest//:gtest",
#         "@com_google_absl//absl/strings:str_format",
#         "//xla:test",
#         "//xla/hlo/testlib:hlo_hardware_independent_test_base",
#         "@local_tsl//tsl/platform:statusor",
#         "@local_tsl//tsl/platform:test_main",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "hlo_element_type_converter",
    srcs = ["simplifiers/hlo_element_type_converter.cc"],
    hdrs = ["simplifiers/hlo_element_type_converter.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "hlo_element_type_converter_test",
    srcs = ["simplifiers/hlo_element_type_converter_test.cc"],
    deps = [
        ":hlo_element_type_converter",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "conditional_canonicalizer",
    srcs = ["simplifiers/conditional_canonicalizer.cc"],
    hdrs = ["simplifiers/conditional_canonicalizer.h"],
    deps = [
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "conditional_canonicalizer_test",
    srcs = ["simplifiers/conditional_canonicalizer_test.cc"],
    deps = [
        ":conditional_canonicalizer",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:literal_test_util",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "zero_sized_hlo_elimination",
    srcs = ["simplifiers/zero_sized_hlo_elimination.cc"],
    hdrs = ["simplifiers/zero_sized_hlo_elimination.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "zero_sized_hlo_elimination_test",
    srcs = ["simplifiers/zero_sized_hlo_elimination_test.cc"],
    deps = [
        ":zero_sized_hlo_elimination",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "sort_simplifier",
    srcs = ["simplifiers/sort_simplifier.cc"],
    hdrs = ["simplifiers/sort_simplifier.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "sort_simplifier_test",
    srcs = ["simplifiers/sort_simplifier_test.cc"],
    deps = [
        ":sort_simplifier",
        "//xla:test",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["expanders/stable_sort_expander.cc"],
    hdrs = ["expanders/stable_sort_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "stable_sort_expander_test",
    srcs = ["expanders/stable_sort_expander_test.cc"],
    deps = [
        ":algebraic_simplifier",
        ":stable_sort_expander",
        "//xla:test",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "root_instruction_sinker",
    srcs = ["simplifiers/root_instruction_sinker.cc"],
    hdrs = ["simplifiers/root_instruction_sinker.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:tuple_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "root_instruction_sinker_test",
    srcs = ["simplifiers/root_instruction_sinker_test.cc"],
    deps = [
        ":root_instruction_sinker",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "convert_memory_placement_to_internal_annotations",
    srcs = ["convert_memory_placement_to_internal_annotations.cc"],
    hdrs = ["convert_memory_placement_to_internal_annotations.h"],
    deps = [
        "//xla:side_effect_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:host_memory_offload_annotations_hdr",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "convert_memory_placement_to_internal_annotations_test",
    srcs = ["convert_memory_placement_to_internal_annotations_test.cc"],
    deps = [
        ":convert_memory_placement_to_internal_annotations",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:host_memory_offload_annotations_hdr",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_memory_transfer_asyncifier",
    srcs = ["simplifiers/host_memory_transfer_asyncifier.cc"],
    hdrs = ["simplifiers/host_memory_transfer_asyncifier.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_memory_transfer_asyncifier_test",
    srcs = ["simplifiers/host_memory_transfer_asyncifier_test.cc"],
    deps = [
        ":host_memory_transfer_asyncifier",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offload_legalize",
    srcs = ["host_offload_legalize.cc"],
    hdrs = ["host_offload_legalize.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:hlo_value",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offload_legalize_test",
    srcs = ["host_offload_legalize_test.cc"],
    shard_count = 12,
    deps = [
        ":host_offload_legalize",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offloader",
    srcs = ["host_offloader.cc"],
    hdrs = ["host_offloader.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_value",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
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
    name = "host_offloader_test",
    srcs = ["host_offloader_test.cc"],
    shard_count = 12,
    deps = [
        ":host_offload_legalize",
        ":host_offloader",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_verifier",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offloading_prepare",
    srcs = ["host_offloading_prepare.cc"],
    hdrs = ["host_offloading_prepare.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:host_memory_offload_annotations_hdr",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offloading_prepare_test",
    srcs = ["host_offloading_prepare_test.cc"],
    deps = [
        ":host_offloading_prepare",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "fusion_constant_sinking",
    srcs = ["simplifiers/fusion_constant_sinking.cc"],
    hdrs = ["simplifiers/fusion_constant_sinking.h"],
    deps = [
        ":hlo_dce",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "fusion_constant_sinking_test",
    srcs = ["simplifiers/fusion_constant_sinking_test.cc"],
    deps = [
        ":fusion_constant_sinking",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "despecializer",
    srcs = ["despecializer.cc"],
    hdrs = ["despecializer.h"],
    deps = [
        ":defuser",
        ":float_normalization",
        ":hlo_memory_scheduler",
        ":sub_byte_normalization",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "literal_canonicalizer",
    srcs = ["literal_canonicalizer.cc"],
    hdrs = ["literal_canonicalizer.h"],
    deps = [
        "//xla:literal_pool",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "literal_canonicalizer_test",
    srcs = ["literal_canonicalizer_test.cc"],
    deps = [
        ":literal_canonicalizer",
        "//xla:literal_pool",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "optimize_input_output_buffer_alias",
    srcs = ["simplifiers/optimize_input_output_buffer_alias.cc"],
    hdrs = ["simplifiers/optimize_input_output_buffer_alias.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "optimize_input_output_buffer_alias_test",
    srcs = ["simplifiers/optimize_input_output_buffer_alias_test.cc"],
    deps = [
        ":optimize_input_output_buffer_alias",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "ar_crs_combiner",
    srcs = ["simplifiers/ar_crs_combiner.cc"],
    hdrs = ["simplifiers/ar_crs_combiner.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_replication_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:call_graph",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["expanders/dynamic_index_splitter.cc"],
    hdrs = ["expanders/dynamic_index_splitter.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "dynamic_index_splitter_test",
    srcs = ["expanders/dynamic_index_splitter_test.cc"],
    deps = [
        ":dynamic_index_splitter",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "ar_crs_combiner_test",
    srcs = ["simplifiers/ar_crs_combiner_test.cc"],
    deps = [
        ":ar_crs_combiner",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "slice_sinker",
    srcs = ["simplifiers/slice_sinker.cc"],
    hdrs = ["simplifiers/slice_sinker.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "slice_sinker_test",
    srcs = ["simplifiers/slice_sinker_test.cc"],
    deps = [
        ":hlo_dce",
        ":slice_sinker",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["expanders/rng_expander.cc"],
    hdrs = ["expanders/rng_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:prng",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "rng_bit_generator_expander",
    srcs = ["expanders/rng_bit_generator_expander.cc"],
    hdrs = ["expanders/rng_bit_generator_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:prng",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "operand_upcaster",
    srcs = ["operand_upcaster.cc"],
    hdrs = ["operand_upcaster.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "operand_upcaster_test",
    srcs = ["operand_upcaster_test.cc"],
    deps = [
        ":operand_upcaster",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "result_caster",
    srcs = ["simplifiers/result_caster.cc"],
    hdrs = ["simplifiers/result_caster.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "result_caster_test",
    srcs = ["simplifiers/result_caster_test.cc"],
    deps = [
        ":result_caster",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "convert_operand_folding",
    srcs = ["simplifiers/convert_operand_folder.cc"],
    hdrs = ["simplifiers/convert_operand_folder.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "convert_operand_folding_test",
    srcs = ["simplifiers/convert_operand_folder_test.cc"],
    deps = [
        ":convert_operand_folding",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "instruction_hoister",
    srcs = ["simplifiers/instruction_hoister.cc"],
    hdrs = ["simplifiers/instruction_hoister.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "gather_simplifier",
    srcs = ["simplifiers/gather_simplifier.cc"],
    hdrs = ["simplifiers/gather_simplifier.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:gather_scatter_utils",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "reduce_window_rewriter",
    srcs = ["simplifiers/reduce_window_rewriter.cc"],
    hdrs = ["simplifiers/reduce_window_rewriter.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
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
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "reduce_window_rewriter_test",
    srcs = ["simplifiers/reduce_window_rewriter_test.cc"],
    deps = [
        ":reduce_window_rewriter",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "stochastic_convert_decomposer",
    srcs = ["expanders/stochastic_convert_decomposer.cc"],
    hdrs = ["expanders/stochastic_convert_decomposer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/container:flat_hash_set",
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
    name = "stochastic_convert_decomposer_test",
    srcs = ["expanders/stochastic_convert_decomposer_test.cc"],
    deps = [
        ":stochastic_convert_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "sub_byte_normalization",
    srcs = ["simplifiers/sub_byte_normalization.cc"],
    hdrs = ["simplifiers/sub_byte_normalization.h"],
    deps = [
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
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
    name = "gather_simplifier_test",
    srcs = ["simplifiers/gather_simplifier_test.cc"],
    deps = [
        ":gather_simplifier",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "add_original_value",
    srcs = ["add_original_value.cc"],
    hdrs = ["add_original_value.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "add_original_value_test",
    srcs = ["add_original_value_test.cc"],
    deps = [
        ":add_original_value",
        "//xla:shape_util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
